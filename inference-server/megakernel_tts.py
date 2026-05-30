"""Megakernel-backed Qwen3-TTS streaming inference wrapper.

This module exposes :class:`MegakernelTTS`, a clean, importable model wrapper
around the three Qwen3-TTS pipeline stages:

    text  ->  Talker (megakernel CUDA decode @ ~13 Hz, vocab 3072)
                 |
                 v
              Code Predictor (5-layer sub-model, 16 codebook groups x vocab 2048)
                 |
                 v
              Codec Decoder (Qwen3-TTS-Tokenizer-12Hz, non-DiT) -> PCM int16 @ 24 kHz

The Talker is intentionally the only stage that runs through AlpinDale's
``qwen_megakernel`` CUDA megakernel; the code predictor and codec decoder are
small enough that running them as plain PyTorch modules on-device is fine and
keeps the integration surface small (matches the brief: "swap megakernel in for
the Talker decode loop only").

The wrapper is async-streaming: :meth:`generate` is an ``AsyncGenerator`` that
yields raw int16 PCM bytes as soon as each codec frame is decoded, so the
caller (Pipecat service, bench harness, raw test script) can begin pushing
audio to its sink with minimal time-to-first-chunk (TTFC).

Status
------
The Talker integration against the modified megakernel ``Decoder`` is
**stubbed** behind ``# TODO: replace with actual megakernel Decoder`` markers.
Wire-up happens after the kernel mods land (MRoPE, hidden=2048, vocab=3072
LM head, untied embeds). Until then, ``generate`` raises
:class:`NotImplementedError` when called with ``stub=False``, and emits zero
PCM in ``stub=True`` mode so the surrounding Pipecat plumbing can be smoke
tested end to end.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

# Qwen3-TTS native codec rate -- confirmed from Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
# config: codec runs at 12.5 Hz frame rate, each frame decodes to 1920 samples
# of 24 kHz int16 PCM (i.e. 80 ms / chunk).
#
# TODO(verify-on-remote): cross-check with codec.config.sampling_rate on the
# loaded model. If Qwen ships a 22.05 kHz variant we surface that here instead.
QWEN3_TTS_SAMPLE_RATE: int = 24_000
QWEN3_TTS_CODEC_FRAME_RATE_HZ: float = 12.5
QWEN3_TTS_SAMPLES_PER_CODEC_FRAME: int = int(
    QWEN3_TTS_SAMPLE_RATE / QWEN3_TTS_CODEC_FRAME_RATE_HZ
)  # 1920

# Talker autoregressive step rate (semantic tokens / second). Used only for
# diagnostics / RTF projection -- the actual decode loop is driven by EOS
# from the talker, not by a fixed token count.
QWEN3_TTS_TALKER_RATE_HZ: float = 13.0


@dataclass
class MegakernelTTSConfig:
    """Configuration for :class:`MegakernelTTS`.

    Parameters:
        model_name: HuggingFace repo id of the Qwen3-TTS checkpoint. The
            canonical target is ``Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice``.
        speaker: Speaker / voice identifier (e.g. ``"ryan"``). Maps to the
            speaker-conditioning embedding consumed by the Talker.
        device: Torch device string. Megakernel hard-requires ``"cuda"``.
        max_new_tokens: Hard cap on Talker steps per utterance. Prevents
            runaway generation if EOS is never emitted.
        sample_rate: Output PCM sample rate. Locked to 24 kHz for Qwen3-TTS;
            exposed here for downstream consumers that prefer reading it off
            the wrapper rather than the module-level constant.
        stub: When True, :meth:`generate` yields silence instead of touching
            the (unwired) Talker. Used for Pipecat plumbing smoke tests
            before kernel mods land. Defaults to False.
    """

    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    speaker: str = "ryan"
    device: str = "cuda"
    max_new_tokens: int = 4096
    sample_rate: int = QWEN3_TTS_SAMPLE_RATE
    stub: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class MegakernelTTS:
    """Streaming Qwen3-TTS pipeline with megakernel-backed Talker decode.

    Lifecycle:
        - ``__init__`` loads the Talker (via the modified megakernel
          ``Decoder``), the code predictor (PyTorch), and the codec decoder
          (PyTorch). All three live on the same CUDA device.
        - :meth:`generate` is invoked per utterance. Internally it runs:
            1. tokenize ``text`` (+ speaker prompt) via the Talker tokenizer
            2. prefill the megakernel KV cache
            3. run an autoregressive Talker decode loop, yielding one semantic
               token at a time
            4. every ~1 semantic token, run the code predictor + codec decoder
               to produce one 80 ms PCM chunk; yield it
            5. stop on EOS or ``max_new_tokens``
        - The class is NOT thread-safe; serialize per-instance access from
          a single asyncio loop.

    The Talker is the ONLY component that goes through the CUDA megakernel.
    Code predictor and codec decoder run as ordinary PyTorch modules; they
    were never the bottleneck and porting them is out of scope for the brief.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        speaker: str = "ryan",
        device: str = "cuda",
        *,
        stub: bool = False,
        config: MegakernelTTSConfig | None = None,
    ) -> None:
        """Initialize the Qwen3-TTS pipeline.

        Args:
            model_name: HuggingFace checkpoint name for Qwen3-TTS.
            speaker: Voice / speaker id for conditioning.
            device: Torch device. Must be ``"cuda"`` for the megakernel path.
            stub: If True, do not touch the Talker on :meth:`generate`; emit
                silence so the surrounding Pipecat pipeline can be smoke tested
                before the kernel mods are wired.
            config: Optional explicit config. If provided, overrides the
                positional args.
        """
        self.config = config or MegakernelTTSConfig(
            model_name=model_name,
            speaker=speaker,
            device=device,
            stub=stub,
        )
        self._sample_rate = self.config.sample_rate

        logger.info(
            "MegakernelTTS init: model={model} speaker={spk} device={dev} stub={stub}",
            model=self.config.model_name,
            spk=self.config.speaker,
            dev=self.config.device,
            stub=self.config.stub,
        )

        # ------------------------------------------------------------------
        # Stage 1: Talker (megakernel CUDA decode)
        # ------------------------------------------------------------------
        # TODO: replace with actual megakernel Decoder once kernel mods land.
        # Expected wiring (post-kernel-mod):
        #
        #   from qwen_megakernel.model import Decoder, load_weights
        #   weights, tokenizer = load_weights(self.config.model_name)
        #   self._talker = Decoder(weights=weights, tokenizer=tokenizer)
        #
        # Kernel mods required (see ~/brain/build/side-projects/
        # project-e3-megakernel-tts.md "Kernel mods needed"):
        #   - hidden=2048 (currently 1024)
        #   - untied lm_head + embed_tokens
        #   - vocab_size=3072 for LM head (semantic codec tokens, not text)
        #   - MRoPE (multi-axis RoPE) instead of vanilla RoPE
        #   - speaker conditioning injection at embed time
        self._talker = None  # type: ignore[assignment]
        self._talker_tokenizer = None  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # Stage 2: Code Predictor (PyTorch, 5 layers, 16 codebook heads)
        # ------------------------------------------------------------------
        # TODO: load from the same HF checkpoint -- usually exposed as
        # ``model.code_predictor`` or a sibling submodule. Confirm exact
        # attribute name from the Qwen3-TTS modeling file on the remote.
        self._code_predictor = None

        # ------------------------------------------------------------------
        # Stage 3: Codec decoder (PyTorch, non-DiT, 12.5 Hz -> 24 kHz PCM)
        # ------------------------------------------------------------------
        # TODO: load Qwen3-TTS-Tokenizer-12Hz codec. Streaming-friendly,
        # accepts one codec frame at a time (16 groups of token ids) and
        # returns ~1920 fp32 samples (or fp16 -- we cast at the yield site).
        self._codec = None

        # ------------------------------------------------------------------
        # Speaker conditioning
        # ------------------------------------------------------------------
        # TODO: resolve self.config.speaker to a speaker-embedding tensor.
        # For ``CustomVoice`` checkpoints the speaker map ships in the repo
        # under ``speaker_embeddings.pt`` (or similar). Cache the resolved
        # tensor on-device so generate() never blocks on a disk read.
        self._speaker_embedding = None

    @property
    def sample_rate(self) -> int:
        """Output PCM sample rate (24 kHz for Qwen3-TTS)."""
        return self._sample_rate

    async def generate(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream PCM audio chunks for ``text``.

        Yields raw int16 little-endian PCM bytes at :attr:`sample_rate` Hz,
        one codec frame (~80 ms of audio) at a time. The first yield happens
        as soon as the codec emits its first frame -- this defines TTFC.

        Args:
            text: The utterance to synthesize. Whitespace-trimmed, no special
                speaker tags required (those come from ``self.config.speaker``).

        Yields:
            ``bytes`` -- a contiguous chunk of int16 LE PCM samples. Roughly
            ``QWEN3_TTS_SAMPLES_PER_CODEC_FRAME * 2`` bytes per yield.

        Raises:
            NotImplementedError: If the Talker is not yet wired and ``stub``
                was not set on the config. Once kernel mods are merged this
                will be a normal generate path.
        """
        text = text.strip()
        if not text:
            return

        if self.config.stub:
            async for chunk in self._stub_generate(text):
                yield chunk
            return

        # TODO: replace with actual megakernel Decoder driven pipeline.
        # Sketch (post-kernel-mod):
        #
        #   self._talker.reset()
        #   prompt_ids = self._talker_tokenizer.encode(text, ...)
        #   for tid in prompt_ids[:-1]:
        #       self._talker.step(tid)               # prefill
        #
        #   semantic_buffer: list[int] = []
        #   next_tok = prompt_ids[-1]
        #   for _ in range(self.config.max_new_tokens):
        #       next_tok = self._talker.step(next_tok)
        #       if next_tok == self._talker_tokenizer.eos_token_id:
        #           break
        #       semantic_buffer.append(next_tok)
        #
        #       # Code predictor expects one Talker token at a time and emits
        #       # 16 codebook ids. Codec needs N codebook frames at minimum
        #       # before it can emit audio -- usually N=1 for the streaming
        #       # variant. Confirm on remote.
        #       code_frame = self._code_predictor.step(next_tok,
        #                                              self._speaker_embedding)
        #       pcm_f32 = self._codec.decode_frame(code_frame)
        #       pcm_i16 = _f32_to_i16_bytes(pcm_f32)
        #       yield pcm_i16
        #       # Cooperatively yield to the event loop so the consumer can
        #       # push to its sink without head-of-line blocking the next
        #       # talker step.
        #       await asyncio.sleep(0)
        raise NotImplementedError(
            "MegakernelTTS.generate(): Talker not yet wired. "
            "Pass stub=True to the constructor for plumbing smoke tests, "
            "or wait for the megakernel kernel-mod merge. "
            "See ~/brain/build/side-projects/project-e3-megakernel-tts.md "
            "'Kernel mods needed' for the wiring checklist."
        )

    async def _stub_generate(self, text: str) -> AsyncGenerator[bytes, None]:
        """Emit silence sized roughly proportional to ``text``.

        Used to smoke-test the Pipecat pipeline before the Talker is wired.
        Yields one 80 ms silent frame per ~3 input characters at 24 kHz int16,
        which is a rough English-prose pacing approximation -- not honest
        synthesis, but enough to drive the downstream audio plumbing.
        """
        logger.warning(
            "MegakernelTTS.generate() running in STUB mode -- emitting silence. "
            "Wire the Talker before benchmarking."
        )
        target_chars_per_frame = 3
        n_frames = max(1, len(text) // target_chars_per_frame)
        silent_frame = np.zeros(
            QWEN3_TTS_SAMPLES_PER_CODEC_FRAME, dtype=np.int16
        ).tobytes()
        for i in range(n_frames):
            yield silent_frame
            # Simulate the ~80ms-per-frame natural pacing. The real path will
            # produce frames much faster than realtime (RTF target < 0.15),
            # so this sleep is stub-only.
            await asyncio.sleep(
                1.0 / QWEN3_TTS_CODEC_FRAME_RATE_HZ
                if self.config.extra.get("stub_realtime", False)
                else 0.0
            )
        del i  # placate linters; loop var unused in body


def _f32_to_i16_bytes(pcm_f32: "np.ndarray") -> bytes:
    """Convert float32 PCM in [-1, 1] to int16 little-endian bytes.

    Centralized so the Pipecat service and the bench harness share the exact
    same quantization. Clips to int16 range to avoid wrap-around on out-of-
    range codec outputs.
    """
    clipped = np.clip(pcm_f32, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def measure_ttfc_ns(start_ns: int) -> int:
    """Return nanoseconds elapsed since ``start_ns`` (perf_counter_ns)."""
    return time.perf_counter_ns() - start_ns


__all__ = [
    "MegakernelTTS",
    "MegakernelTTSConfig",
    "QWEN3_TTS_CODEC_FRAME_RATE_HZ",
    "QWEN3_TTS_SAMPLE_RATE",
    "QWEN3_TTS_SAMPLES_PER_CODEC_FRAME",
    "QWEN3_TTS_TALKER_RATE_HZ",
    "measure_ttfc_ns",
]
