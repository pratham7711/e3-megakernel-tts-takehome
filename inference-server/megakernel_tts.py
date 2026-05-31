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

# Qwen3-TTS audio-side codec token ids (verified against the upstream
# Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice config.json + the QwenLM/Qwen3-TTS
# inference reference at qwen_tts/core/models/modeling_qwen3_tts.py:1240-1295).
# These IDs live in the audio vocab (size 3072), NOT the text vocab.
QWEN3_TTS_CODEC_PAD_TOKEN_ID: int = 2148      # codec_pad_id
QWEN3_TTS_CODEC_BOS_TOKEN_ID: int = 2149      # codec_bos_id  — the "begin generating audio" sentinel
QWEN3_TTS_CODEC_EOS_TOKEN_ID: int = 2150      # codec_eos_token_id
QWEN3_TTS_CODEC_THINK_ID: int = 2154          # codec_think_id
QWEN3_TTS_CODEC_THINK_BOS_ID: int = 2156      # codec_think_bos_id
QWEN3_TTS_CODEC_THINK_EOS_ID: int = 2157      # codec_think_eos_id
QWEN3_TTS_LANGUAGE_ENGLISH: int = 2050        # codec_language_id["english"]

# Speaker id map — talker_config.spk_id from upstream config.json (range
# 2861-3066 in the audio vocab). Reading at __init__ time from the on-disk
# config would be more robust; we hard-code here for the canonical 9 voices.
QWEN3_TTS_SPEAKER_IDS: dict[str, int] = {
    "serena":   2861,
    "vivian":   2862,
    "uncle_fu": 2863,
    "ryan":     3061,
    "aiden":    3062,
    "ono_anna": 3063,
    "sohee":    3064,
    "eric":     3065,
    "dylan":    3066,
}

# Audio-side prefix that the upstream model injects between the text prefill
# and the AR audio decode. ORDER MATTERS — derived from
# modeling_qwen3_tts.py:1240-1276. We prefill the first 6 of these and let
# the AR loop's FIRST step() seed codec_bos itself (so the kernel's
# step(prev_tok=2149) writes codec_bos at the right position and predicts
# the first audio token from that state, matching upstream semantics).
def _audio_prefix_token_ids(spk_id: int, language_id: int = QWEN3_TTS_LANGUAGE_ENGLISH) -> list[int]:
    return [
        QWEN3_TTS_CODEC_THINK_ID,        # 2154
        QWEN3_TTS_CODEC_THINK_BOS_ID,    # 2156
        language_id,                      # 2050 for English
        QWEN3_TTS_CODEC_THINK_EOS_ID,    # 2157
        spk_id,                           # e.g. 3061 for ryan
        QWEN3_TTS_CODEC_PAD_TOKEN_ID,    # 2148
        # NOTE: codec_bos (2149) is fed via the FIRST step() call, not here
    ]


def _resolve_speaker_id(speaker: str) -> int:
    """Resolve a speaker name (e.g. 'ryan') to its codec token id."""
    key = speaker.lower().strip()
    if key not in QWEN3_TTS_SPEAKER_IDS:
        raise ValueError(
            f"Unknown speaker {speaker!r}. Valid speakers: {sorted(QWEN3_TTS_SPEAKER_IDS)}"
        )
    return QWEN3_TTS_SPEAKER_IDS[key]


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
    compile_per_frame: bool = field(
        default_factory=lambda: os.environ.get("MEGAKERNEL_COMPILE_PER_FRAME", "1") == "1"
    )
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

        # Warm up RoPE tables on the target device/dtype BEFORE torch.compile
        # traces. If the tables get materialised inside the compiled function,
        # they end up in the CUDA-graph private pool and the second compiled
        # call hits a "storage overwritten" runtime error. Building them
        # eagerly here places them in the normal allocator pool.
        target_device = torch.device(self.config.device)
        target_dtype = torch.bfloat16
        self._code_predictor.warmup_rope(target_device, target_dtype)
        if hasattr(self._codec, "pre_transformer") and self._codec.pre_transformer is not None:
            self._codec.pre_transformer.warmup_rope(target_device, target_dtype)

        # Trigger torch.compile graph capture + prefill_text warmup NOW so the
        # first run_tts() call in production doesn't pay the cold-start cost.
        # Bench harness gets this for free via its 3 warmup generate() calls;
        # Pipecat e2e is a one-shot, so without this the canonical
        # UserStopped → BotStarted latency would be dominated by compile time.
        if self.config.compile_per_frame:
            try:
                logger.info(
                    "MegakernelTTS: pre-warming compile + prefill_text "
                    "(this takes ~20-25 s at model load, amortizes per-turn)…"
                )
                t0 = time.perf_counter()
                # Phase 1: per-frame fn compile + CUDA graph capture (3 iters,
                # token id 0). ``_call`` handles cudagraph_mark_step_begin.
                per_frame = self._build_per_frame_fn()
                for _ in range(3):
                    _ = per_frame(0)
                logger.info(
                    "MegakernelTTS: per-frame compile warmup done in {s:.1f} s",
                    s=time.perf_counter() - t0,
                )
                # Phase 2: prefill_text warmup. The talker's prefill_text
                # triggers a PyTorch forward through the text-projection
                # submodule on first call — ~1.5 s cold, <50 ms warm. Do it
                # now under a dummy utterance so the user's first turn is fast.
                t1 = time.perf_counter()
                try:
                    self._talker.prefill_text("Hello there.", model_path=self.config.model_path)
                    self._talker.reset()
                    logger.info(
                        "MegakernelTTS: prefill_text warmup done in {s:.2f} s",
                        s=time.perf_counter() - t1,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "MegakernelTTS: prefill_text warmup failed ({err!r}); "
                        "first generate() call will pay the prefill compile cost.",
                        err=e,
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "MegakernelTTS: compile warmup failed ({err!r}); first "
                    "run_tts() will pay the compile cost.",
                    err=e,
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

        # When the compiled path uses reduce-overhead (CUDA graphs), PyTorch
        # reuses output storage across calls — meaning the previous call's
        # output gets overwritten when the next call runs. We have to mark
        # a "step begin" between invocations so the framework treats each
        # frame as a separate iteration. No-op when torch.compile fell back.
        _cudagraph_mark = getattr(
            getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None
        )

        def _call(token_id: int) -> bytes:
            if _cudagraph_mark is not None and self.config.compile_per_frame:
                _cudagraph_mark()
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
            # Silence stub is plumbing-only — does NOT meet the brief.
            # The deliverable IS the megakernel Qwen3-TTS path below.
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

        # Single-pass combined prefill matching the upstream Qwen3-TTS flow
        # (modeling_qwen3_tts.py:1240-1295). The text (wrapped in the
        # `<tts_text_bos><|im_start|>assistant\n…<|im_end|>\n<|im_start|>
        # assistant\n<tts_text_eod>` template, projected through
        # text_embedding + text_projection MLP) is CONCATENATED with the
        # 6-token codec-side prefix
        #   [codec_think, codec_think_bos, language=english, codec_think_eos,
        #    speaker_embed(spk_id), codec_pad]
        # into a SINGLE embedding sequence. The 28-layer prefill forward
        # then runs ONCE, so audio-prefix tokens see the text K/V within
        # their own self-attention causal window — exactly matching the
        # upstream `talker.generate(inputs_embeds=cat([text, codec_pref]))`
        # semantics. The first AR step is seeded with codec_bos (2149).
        do_prefill = (
            self.config.text_prefill if text_prefill is None else bool(text_prefill)
        )
        n_total = 0
        if do_prefill:
            try:
                spk_id = _resolve_speaker_id(self.config.speaker)
                prefix_ids = _audio_prefix_token_ids(spk_id, QWEN3_TTS_LANGUAGE_ENGLISH)
                n_total = self._talker.prefill_text(
                    text,
                    model_path=self.config.model_path,
                    codec_prefix_ids=prefix_ids,
                )
                logger.debug(
                    "MegakernelTTS: combined prefill wrote {n} tokens to KV "
                    "cache (text + 6 audio-prefix; speaker={sp} -> spk_id={sid}).",
                    n=n_total, sp=self.config.speaker, sid=spk_id,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "MegakernelTTS: combined prefill failed ({err!r}); "
                    "proceeding without conditioning (audio will be unconditioned).",
                    err=e,
                )

        # First AR step seeds codec_bos (2149) — the upstream "begin generating
        # audio" sentinel. The kernel writes its embedding at the next position
        # and predicts the first audio token from that state.
        prev_tok: int = QWEN3_TTS_CODEC_BOS_TOKEN_ID if do_prefill else 0

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
            # NOTE: no asyncio.sleep(0) here. The `yield` above is already an
            # async-generator yield point — control returns to the consumer
            # without an extra reschedule. sleep(0) added 50-200 µs per frame
            # (~3-13 ms per 5-sec utterance) with no head-of-line benefit
            # because Pipecat drains at codec rate (~80 ms gap), far slower
            # than our produce rate. Do NOT reintroduce.

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
