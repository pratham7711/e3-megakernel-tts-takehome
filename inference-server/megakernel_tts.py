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
import os
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
    # Whether to run text prefill before the autoregressive audio decode.
    # Default True: produces intelligible English audio conditioned on the
    # input text. Set False (e.g. from the talker-only bench) to measure
    # raw decode throughput without paying the prefill cost.
    text_prefill: bool = True
    # Compile the per-frame code_predictor + codec into a single graphed
    # callable (torch.compile with reduce-overhead mode). This amortizes
    # the ~600x per-frame CUDA dispatch overhead that we observed when
    # running the un-fused per-frame path naively. The first call pays
    # the compile cost; subsequent calls reuse the compiled artifact.
    compile_per_frame: bool = True
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
        # Compiled per-frame callable: takes a Python int talker token id,
        # returns int16 PCM bytes for that ONE codec frame. Populated lazily
        # on the first generate() call so we know which device/dtype to bind.
        self._per_frame_fn = None
        self._per_frame_warmed = False

        if self.config.stub:
            logger.warning(
                "MegakernelTTS initialised in STUB mode -- skipping Decoder/"
                "code_predictor/codec load. generate() will emit silence."
            )
            return

        # Real init path. The caller asked for the real megakernel — if any
        # stage fails to load, RAISE so the surrounding pipeline fails loud.
        # Silent fallback to stub once masked a regression where the file-mode
        # smoke test "passed" but actually recorded the user's input echoed
        # back. If you need silence-only plumbing tests, pass stub=True
        # explicitly.
        from qwen_megakernel.model import Decoder  # type: ignore

        self._talker = Decoder(
            model_path=self.config.model_path,
            verbose=False,
        )
        logger.info("MegakernelTTS: Decoder loaded from {p}", p=self.config.model_path)

        from qwen3_tts_components import load_components  # type: ignore
        import torch  # type: ignore

        # H1 fix: load_components() now returns 3-tuple (cp, codec, info)
        # after the real-codec rewrite. Unpacking 2 silently flipped the
        # service into STUB mode.
        self._code_predictor, self._codec, _info = load_components(
            weights_dir=self.config.model_path,
            device=self.config.device,
            dtype=torch.bfloat16,
        )
        logger.info(
            "MegakernelTTS: code_predictor + codec loaded via "
            "qwen3_tts_components.load_components()"
        )

    @property
    def sample_rate(self) -> int:
        """Output PCM sample rate (24 kHz for Qwen3-TTS)."""
        return self._sample_rate

    def _build_per_frame_fn(self):
        """Build (and lazily compile) the per-frame talker_token_id -> PCM-bytes fn.

        This is the streaming hot path: ONE talker token in, ONE codec
        frame's worth of int16 PCM bytes out, with zero batching across
        frames. Naively per-frame this is ~600x slower than the batched
        path we used pre-streaming because each frame pays the full CUDA
        dispatch + Python overhead for two PyTorch modules.

        Mitigation: wrap the inner ``code_predictor + codec`` pair in
        ``torch.compile(mode="reduce-overhead")``. reduce-overhead enables
        CUDA Graph capture under the hood when the input shape is stable
        (here, a constant ``(1, 1)`` LongTensor) so each call replays a
        pre-captured graph instead of re-issuing per-op CUDA launches.

        If torch.compile fails (no triton, old torch, etc.) we transparently
        fall back to the raw uncompiled path so the pipeline still works
        end-to-end -- it will just be slower per-frame.
        """
        if self._per_frame_fn is not None:
            return self._per_frame_fn

        import torch  # type: ignore

        code_predictor = self._code_predictor
        codec = self._codec
        device = self.config.device

        # The actual per-frame body. Pure tensor ops, no Python branching
        # inside, fixed shapes -- ideal for torch.compile / cuda graph.
        @torch.no_grad()
        def _raw(tok_tensor: "torch.Tensor") -> "torch.Tensor":
            code_frame = code_predictor(tok_tensor)  # (1, 1, 16) long
            pcm = codec(code_frame)                  # (1, 1920) float OR int16
            return pcm

        compiled = _raw
        if self.config.compile_per_frame:
            try:
                compiled = torch.compile(_raw, mode="reduce-overhead", dynamic=False)
                logger.info(
                    "MegakernelTTS: per-frame fn wrapped in "
                    "torch.compile(mode='reduce-overhead')."
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "MegakernelTTS: torch.compile failed for per-frame fn "
                    "({err!r}); falling back to raw eager path.",
                    err=e,
                )
                compiled = _raw

        def _call(token_id: int) -> bytes:
            tok_tensor = torch.tensor(
                [[token_id]], dtype=torch.long, device=device
            )
            pcm = compiled(tok_tensor)
            if isinstance(pcm, (bytes, bytearray)):
                return bytes(pcm)
            if isinstance(pcm, np.ndarray):
                return pcm.astype(np.int16).tobytes()
            # torch.Tensor: float in [-1, 1] OR already int16. Reshape to 1D.
            t = pcm.detach()
            if t.dtype == torch.int16:
                return t.to("cpu").contiguous().view(-1).numpy().tobytes()
            # Float audio -- clip and quantize. fp32 on host for accuracy.
            t = t.to("cpu").to(torch.float32).view(-1).numpy()
            t = np.clip(t, -1.0, 1.0)
            return (t * 32767.0).astype(np.int16).tobytes()

        self._per_frame_fn = _call
        return _call

    async def generate(
        self,
        text: str,
        *,
        text_prefill: bool | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream PCM audio chunks for ``text``.

        Yields raw int16 little-endian PCM bytes at :attr:`sample_rate` Hz,
        one codec frame (~80 ms of audio) at a time, EMITTED IMMEDIATELY as
        each frame is decoded -- no end-of-utterance buffering. The first
        yield happens as soon as the codec emits its first frame; this
        defines TTFC and satisfies the brief's "push audio chunks as they're
        decoded, do NOT buffer the full utterance before sending" requirement.

        Args:
            text: The utterance to synthesize. Whitespace-trimmed, no special
                speaker tags required (those come from ``self.config.speaker``).
            text_prefill: Override ``self.config.text_prefill`` for this call.
                Default ``None`` -> use the config value. Bench harnesses
                pass ``False`` to measure pure-decode throughput.

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
            stub_mode = self.config.extra.get("stub_mode", "silence")
            if stub_mode == "mac_say":
                async for chunk in self._mac_say_generate(text):
                    yield chunk
            else:
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

        # Run text prefill so the audio is intelligibly conditioned on the
        # input string. The Decoder.prefill_text() method does this in pure
        # PyTorch using the same weight tensors the kernel uses, and writes
        # K/V directly into the megakernel's KV cache buffers. After this
        # call, self._talker._position is set to the prefill length and the
        # following step() calls continue autoregressively.
        do_prefill = (
            self.config.text_prefill if text_prefill is None else bool(text_prefill)
        )
        if do_prefill:
            try:
                n = self._talker.prefill_text(text, model_path=self.config.model_path)
                logger.debug(
                    "MegakernelTTS: text prefill wrote {n} tokens to KV cache.",
                    n=n,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "MegakernelTTS: prefill_text() failed ({err!r}); "
                    "proceeding without prefill (audio will be unconditioned).",
                    err=e,
                )

        # Seed the first audio decode step. With prefill the talker has
        # already absorbed the text context, so token 0 is just the
        # "begin audio" sentinel; without prefill we fall back to the
        # pre-prefill behaviour (unconditioned audio).
        prev_tok: int = 0

        max_new = self.config.max_new_tokens
        eos = self.config.eos_token_id

        per_frame = self._build_per_frame_fn()

        for _ in range(max_new):
            next_tok = self._talker.step(prev_tok)
            if next_tok == eos:
                break

            # Run code_predictor + codec on this ONE frame and emit
            # immediately. This is the streaming yield required by the brief.
            pcm_bytes = per_frame(next_tok)
            yield pcm_bytes

            prev_tok = next_tok

            # Cooperatively yield to the event loop so the consumer can push
            # to its sink without head-of-line blocking the next talker step.
            await asyncio.sleep(0)

    async def _mac_say_generate(self, text: str) -> AsyncGenerator[bytes, None]:
        """Mac-only stub: synthesize ``text`` via macOS ``say`` and stream PCM.

        Lets us validate the end-to-end Pipecat pipeline on a laptop without
        CUDA: real speech audio at 24 kHz mono int16, chunked into 80 ms
        frames that match the codec's native frame size, so the downstream
        BotAudioRecorder / LocalAudioOutputTransport sees frames identical
        in shape to what the real megakernel TTS will eventually emit.
        """
        import subprocess
        import tempfile

        logger.warning(
            "MegakernelTTS.generate() running in STUB mode=mac_say — using "
            "macOS `say` as a voice substitute. Real megakernel runs on CUDA."
        )

        aiff_fd, aiff = tempfile.mkstemp(suffix=".aiff")
        os.close(aiff_fd)
        wav_fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)

        try:
            subprocess.run(["say", "-o", aiff, text], check=True, timeout=30)
            subprocess.run(
                [
                    "afconvert",
                    "-f", "WAVE",
                    "-d", f"LEI16@{QWEN3_TTS_SAMPLE_RATE}",
                    "-c", "1",
                    aiff, wav,
                ],
                check=True,
                timeout=20,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("mac_say stub failed: {e!r}", e=e)
            try:
                os.unlink(aiff)
                os.unlink(wav)
            except OSError:
                pass
            return

        with open(wav, "rb") as f:
            # Strip the 44-byte WAV header so we yield raw PCM bytes (same
            # shape downstream consumers expect from the real codec).
            f.read(44)
            audio_bytes = f.read()
        try:
            os.unlink(aiff)
            os.unlink(wav)
        except OSError:
            pass

        # Chunk into codec-frame-sized pieces so streaming behaviour
        # mirrors the real path. 1920 samples * 2 bytes = 3840 bytes/frame.
        bytes_per_frame = QWEN3_TTS_SAMPLES_PER_CODEC_FRAME * 2
        n_frames = (len(audio_bytes) + bytes_per_frame - 1) // bytes_per_frame
        logger.info(
            "mac_say: {chars} chars -> {bytes} bytes -> {n} 80 ms frames",
            chars=len(text),
            bytes=len(audio_bytes),
            n=n_frames,
        )
        for i in range(n_frames):
            chunk = audio_bytes[i * bytes_per_frame:(i + 1) * bytes_per_frame]
            if len(chunk) < bytes_per_frame:
                # Pad the tail frame to full size so downstream framing
                # assumptions hold.
                chunk = chunk + b"\x00" * (bytes_per_frame - len(chunk))
            yield chunk
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
