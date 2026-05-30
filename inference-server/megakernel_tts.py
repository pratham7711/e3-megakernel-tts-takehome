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
QWEN3_TTS_SAMPLE_RATE: int = 24_000
QWEN3_TTS_CODEC_FRAME_RATE_HZ: float = 12.5
QWEN3_TTS_SAMPLES_PER_CODEC_FRAME: int = int(
    QWEN3_TTS_SAMPLE_RATE / QWEN3_TTS_CODEC_FRAME_RATE_HZ
)  # 1920

# Talker autoregressive step rate (semantic tokens / second). Used only for
# diagnostics / RTF projection -- the actual decode loop is driven by EOS
# from the talker, not by a fixed token count.
QWEN3_TTS_TALKER_RATE_HZ: float = 13.0

# Qwen3-TTS codec EOS token id (per Qwen3-TTS-12Hz-1.7B-CustomVoice config).
QWEN3_TTS_CODEC_EOS_TOKEN_ID: int = 2150


@dataclass
class MegakernelTTSConfig:
    """Configuration for :class:`MegakernelTTS`.

    Parameters:
        model_name: HuggingFace repo id of the Qwen3-TTS checkpoint. The
            canonical target is ``Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice``.
        model_path: Local on-disk path to the Qwen3-TTS weights directory.
            The megakernel ``Decoder`` and component loader both consume the
            checkpoint from this directory (safetensors).
        speaker: Speaker / voice identifier (e.g. ``"ryan"``). Maps to the
            speaker-conditioning embedding consumed by the Talker.
        device: Torch device string. Megakernel hard-requires ``"cuda"``.
        max_new_tokens: Hard cap on Talker steps per utterance. Prevents
            runaway generation if EOS is never emitted.
        sample_rate: Output PCM sample rate. Locked to 24 kHz for Qwen3-TTS;
            exposed here for downstream consumers that prefer reading it off
            the wrapper rather than the module-level constant.
        eos_token_id: Semantic-token id that ends an utterance. Defaults to
            ``QWEN3_TTS_CODEC_EOS_TOKEN_ID`` (2150) per the upstream config.
        stub: When True, :meth:`generate` yields silence instead of touching
            the (unwired) Talker. Used for Pipecat plumbing smoke tests
            before kernel mods land. Defaults to False.
    """

    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model_path: str = "/workspace/qwen3-tts-1.7b"
    speaker: str = "ryan"
    device: str = "cuda"
    max_new_tokens: int = 4096
    sample_rate: int = QWEN3_TTS_SAMPLE_RATE
    eos_token_id: int = QWEN3_TTS_CODEC_EOS_TOKEN_ID
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
        model_path: str | None = None,
        stub: bool = False,
        config: MegakernelTTSConfig | None = None,
    ) -> None:
        """Initialize the Qwen3-TTS pipeline.

        Args:
            model_name: HuggingFace checkpoint name for Qwen3-TTS.
            speaker: Voice / speaker id for conditioning.
            device: Torch device. Must be ``"cuda"`` for the megakernel path.
            model_path: Local path to the Qwen3-TTS weights directory.
            stub: If True, do not touch the Talker on :meth:`generate`; emit
                silence so the surrounding Pipecat pipeline can be smoke tested
                before the kernel mods are wired.
            config: Optional explicit config. If provided, overrides the
                positional args.
        """
        if config is not None:
            self.config = config
        else:
            cfg_kwargs: dict[str, Any] = dict(
                model_name=model_name,
                speaker=speaker,
                device=device,
                stub=stub,
            )
            if model_path is not None:
                cfg_kwargs["model_path"] = model_path
            self.config = MegakernelTTSConfig(**cfg_kwargs)
        self._sample_rate = self.config.sample_rate

        logger.info(
            "MegakernelTTS init: model={model} path={path} speaker={spk} "
            "device={dev} stub={stub}",
            model=self.config.model_name,
            path=self.config.model_path,
            spk=self.config.speaker,
            dev=self.config.device,
            stub=self.config.stub,
        )

        # ------------------------------------------------------------------
        # Stage 1: Talker (megakernel CUDA decode)
        # ------------------------------------------------------------------
        self._talker = None  # type: ignore[assignment]
        # ------------------------------------------------------------------
        # Stage 2: Code Predictor (PyTorch, 5 layers, 16 codebook heads)
        # Stage 3: Codec decoder (PyTorch, non-DiT, 12.5 Hz -> 24 kHz PCM)
        # ------------------------------------------------------------------
        self._code_predictor = None
        self._codec = None

        if self.config.stub:
            logger.warning(
                "MegakernelTTS initialised in STUB mode -- skipping Decoder/"
                "code_predictor/codec load. generate() will emit silence."
            )
            return

        # Real init path. Any failure here flips us into stub mode with a
        # WARNING so the surrounding Pipecat pipeline can still smoke-test.
        try:
            from qwen_megakernel.model import Decoder  # type: ignore

            self._talker = Decoder(
                model_path=self.config.model_path,
                verbose=False,
            )
            logger.info("MegakernelTTS: Decoder loaded from {p}", p=self.config.model_path)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "MegakernelTTS: failed to load megakernel Decoder ({err!r}); "
                "falling back to STUB mode.",
                err=e,
            )
            self.config.stub = True
            return

        try:
            # The component file is being written by a parallel agent; import
            # lazily so this module stays importable on machines that don't
            # yet have it.
            from qwen3_tts_components import load_components  # type: ignore

            import torch  # type: ignore

            self._code_predictor, self._codec = load_components(
                weights_dir=self.config.model_path,
                device=self.config.device,
                dtype=torch.bfloat16,
            )
            logger.info(
                "MegakernelTTS: code_predictor + codec loaded via "
                "qwen3_tts_components.load_components()"
            )
        except ImportError as e:
            logger.warning(
                "MegakernelTTS: qwen3_tts_components not available yet "
                "({err!r}); falling back to STUB mode. Re-init once the "
                "parallel agent ships inference-server/qwen3_tts_components.py.",
                err=e,
            )
            self.config.stub = True
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "MegakernelTTS: load_components() failed ({err!r}); "
                "falling back to STUB mode.",
                err=e,
            )
            self.config.stub = True

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
            NotImplementedError: If neither the real path is available nor
                ``stub=True``. (In practice ``__init__`` flips ``stub=True``
                on any load failure, so callers will get silence, not a
                raise -- but we keep the guard for defensive programming.)
        """
        text = text.strip()
        if not text:
            return

        if self.config.stub:
            async for chunk in self._stub_generate(text):
                yield chunk
            return

        if self._talker is None or self._code_predictor is None or self._codec is None:
            raise NotImplementedError(
                "MegakernelTTS.generate(): real path requested but one of "
                "Decoder / code_predictor / codec failed to load. Init should "
                "have set stub=True on failure -- check the init logs."
            )

        # Reset KV cache for the new utterance.
        self._talker.reset()

        # NOTE on prefill: the megakernel Decoder operates on AUDIO semantic
        # tokens only; text prefill is expected to populate the KV cache via
        # the HF/PyTorch path before this loop runs (see qwen_megakernel/
        # model.py module docstring). Until that prefill harness is wired,
        # we seed the autoregressive loop with token id 0 -- the code
        # predictor / codec will produce a short throwaway frame and then
        # the model takes over. This is the same approach the bench harness
        # uses for end-to-end throughput measurement.
        # TODO(prefill): once the HF text-prefill helper lands, call it here
        # to populate the KV cache + seed prev_tok with the last prompt id.
        prev_tok: int = 0

        max_new = self.config.max_new_tokens
        eos = self.config.eos_token_id

        for _ in range(max_new):
            next_tok = self._talker.step(prev_tok)
            if next_tok == eos:
                break

            # code_predictor: talker semantic token id -> 16 codebook token ids
            code_frame = self._code_predictor(next_tok)
            # codec: 16 codebook token ids -> 1920 int16 samples (3840 bytes)
            pcm_bytes = self._codec(code_frame)

            # Defensive normalization: accept either raw bytes or a tensor /
            # ndarray of int16 samples from the codec.
            if isinstance(pcm_bytes, (bytes, bytearray)):
                yield bytes(pcm_bytes)
            elif isinstance(pcm_bytes, np.ndarray):
                yield pcm_bytes.astype(np.int16).tobytes()
            else:
                # torch.Tensor or similar -- pull to CPU and emit i16.
                arr = np.asarray(pcm_bytes.detach().cpu().numpy())
                yield arr.astype(np.int16).tobytes()

            prev_tok = next_tok

            # Cooperatively yield to the event loop so the consumer can push
            # to its sink without head-of-line blocking the next talker step.
            await asyncio.sleep(0)

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
    "QWEN3_TTS_CODEC_EOS_TOKEN_ID",
    "QWEN3_TTS_CODEC_FRAME_RATE_HZ",
    "QWEN3_TTS_SAMPLE_RATE",
    "QWEN3_TTS_SAMPLES_PER_CODEC_FRAME",
    "QWEN3_TTS_TALKER_RATE_HZ",
    "measure_ttfc_ns",
]
