"""Pipecat TTSService backed by :mod:`megakernel_tts`.

Wraps :class:`megakernel_tts.MegakernelTTS` -- the Qwen3-TTS pipeline whose
Talker decode runs through AlpinDale's CUDA megakernel -- as a Pipecat
``TTSService`` so it can drop into any Pipecat voice pipeline in place of
Deepgram/Cartesia/ElevenLabs/Kokoro.

Modeled on ``pipecat.services.kokoro.tts.KokoroTTSService`` because that's
the closest streaming-local-model template in tree: both wrap a Python model
object that yields audio chunks, both run fully on-device, neither needs an
API key. Differences vs. Kokoro:

- We expose ``model_name`` / ``speaker`` / ``device`` directly in ``__init__``
  rather than going through the deprecated ``params``/``settings`` dance
  -- this is a take-home submission and there are no legacy callers to
  protect.
- Qwen3-TTS is fixed at 24 kHz so we skip stream-resampling unless the output
  transport explicitly wants a different ``audio_out_sample_rate`` at
  ``StartFrame`` time.
- No ``ErrorFrame`` swallowing -- exceptions surface so the bench harness
  fails loud.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator

from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

from megakernel_tts import (
    QWEN3_TTS_SAMPLE_RATE,
    MegakernelTTS,
    MegakernelTTSConfig,
)


class MegakernelTTSService(TTSService):
    """Pipecat TTSService that streams audio from a local Qwen3-TTS pipeline.

    The Talker decode runs through the modified ``qwen_megakernel`` CUDA
    megakernel; the code predictor and codec decoder are vanilla PyTorch on
    the same GPU. The service produces 16-bit mono PCM at 24 kHz (Qwen3-TTS
    native rate). Pipecat resamples to ``audio_out_sample_rate`` if the
    downstream transport requested something different at ``StartFrame``.

    Metrics:
        - TTFB (time-to-first-byte) is reported via the standard
          ``self.stop_ttfb_metrics()`` hook the moment the first PCM chunk is
          yielded -- this is the canonical Pipecat surrogate for "TTFC".
        - Usage metrics (chars/words) are reported via
          ``self.start_tts_usage_metrics(text)``.
    """

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        model_path: str = "/workspace/qwen3-tts-1.7b",
        speaker: str = "ryan",
        device: str = "cuda",
        stub: bool = False,
        sample_rate: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize the megakernel-backed TTS service.

        Args:
            model_name: HuggingFace Qwen3-TTS checkpoint id. Defaults to the
                canonical ``Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice``.
            model_path: Local on-disk path to the Qwen3-TTS weights directory.
                Used by the megakernel ``Decoder`` and the code_predictor +
                codec loader.
            speaker: Speaker / voice identifier, e.g. ``"ryan"``. Resolved to
                a speaker-embedding tensor inside :class:`MegakernelTTS`.
            device: Torch device. Must be ``"cuda"`` for the megakernel path.
            stub: If True, the underlying :class:`MegakernelTTS` emits silence
                instead of touching the (unwired) Talker. Use for end-to-end
                Pipecat plumbing smoke tests before kernel mods land.
            sample_rate: Override the output sample rate. Defaults to 24 kHz
                (Qwen3-TTS native). Leave as None unless you have a reason.
            **kwargs: Forwarded to :class:`pipecat.services.tts_service.TTSService`.
        """
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate or QWEN3_TTS_SAMPLE_RATE,
            # Initialise stock TTSSettings fields so Pipecat's validator
            # doesn't log a NOT_GIVEN error every run. We don't honor remote
            # voice / model overrides — those would re-route around the
            # megakernel — so None is the honest answer.
            model=model_name,
            voice=speaker,
            language=None,
            **kwargs,
        )

        self._model_name = model_name
        self._model_path = model_path
        self._speaker = speaker
        self._device = device

        config = MegakernelTTSConfig(
            model_name=model_name,
            model_path=model_path,
            speaker=speaker,
            device=device,
            stub=stub,
        )
        self._tts = MegakernelTTS(config=config)

        # Source sample rate is the model's native rate. If the pipeline asks
        # us to ship a different ``audio_out_sample_rate`` we resample on the
        # fly using Pipecat's stream resampler (same pattern as Kokoro).
        self._source_sample_rate = self._tts.sample_rate
        self._resampler = create_stream_resampler()

        logger.info(
            "MegakernelTTSService ready: model={m} speaker={s} device={d} "
            "source_rate={sr} stub={stub}",
            m=model_name,
            s=speaker,
            d=device,
            sr=self._source_sample_rate,
            stub=stub,
        )

    def can_generate_metrics(self) -> bool:
        """Indicate that this service supports TTFB + usage metrics."""
        return True

    async def stream_tts(
        self,
        text: str,
        *,
        max_new_tokens: int | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Public streaming wrapper around the underlying megakernel.

        Yields raw int16 LE PCM ``bytes`` at :attr:`sample_rate` Hz, one codec
        frame (~80 ms) at a time. Use this from non-Pipecat callers (e.g. the
        Gradio UI) so they don't have to reach through the ``_tts`` private
        attribute. The Pipecat ``run_tts`` path above is unchanged.

        Args:
            text: Utterance to synthesize.
            max_new_tokens: Optional hard cap on Talker decode steps. ``None``
                falls back to ``MegakernelTTSConfig.max_new_tokens``.

        Yields:
            ``bytes`` — one codec frame of int16 LE PCM per yield.
        """
        async for pcm_bytes in self._tts.generate(
            text, max_new_tokens=max_new_tokens
        ):
            yield pcm_bytes

    @property
    def source_sample_rate(self) -> int:
        """Native sample rate of the underlying megakernel (24 kHz)."""
        return self._source_sample_rate

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize ``text`` and yield ``TTSAudioRawFrame`` per chunk.

        Drives the underlying :class:`MegakernelTTS` async generator. The
        first ``TTSAudioRawFrame`` is yielded as soon as the codec emits its
        first frame; subsequent frames stream at codec rate (~12.5 frames/s
        of speech).

        Args:
            text: Text to synthesize. Pipecat may have already aggregated
                multiple sentence-fragment ``TextFrame`` s into ``text``.
            context_id: Per-utterance context id; propagated onto every
                ``TTSAudioRawFrame`` so the output transport can route /
                interrupt correctly.

        Yields:
            ``TTSAudioRawFrame`` for each codec frame, plus a terminal
            ``ErrorFrame`` if synthesis fails.
        """
        logger.debug("MegakernelTTSService: generating TTS [{t!r}]", t=text)

        t0_ns = time.perf_counter_ns()
        first_chunk = True

        try:
            await self.start_tts_usage_metrics(text)

            async for pcm_bytes in self._tts.generate(text):
                if first_chunk:
                    await self.stop_ttfb_metrics()
                    ttfc_ms = (time.perf_counter_ns() - t0_ns) / 1_000_000.0
                    logger.info(
                        "MegakernelTTSService TTFC={ms:.1f} ms (context={cid})",
                        ms=ttfc_ms,
                        cid=context_id,
                    )
                    first_chunk = False

                # Resample only if the pipeline asked for a non-native rate.
                if self._source_sample_rate != self.sample_rate:
                    audio = await self._resampler.resample(
                        pcm_bytes,
                        self._source_sample_rate,
                        self.sample_rate,
                    )
                else:
                    audio = pcm_bytes

                yield TTSAudioRawFrame(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as e:  # noqa: BLE001 -- TTSService convention is to push ErrorFrame
            logger.exception("MegakernelTTSService: synthesis failed")
            yield ErrorFrame(error=f"MegakernelTTSService error: {e}")
        finally:
            # Idempotent; safe to call even if first_chunk never flipped.
            await self.stop_ttfb_metrics()


__all__ = ["MegakernelTTSService"]
