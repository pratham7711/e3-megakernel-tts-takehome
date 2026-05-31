"""End-to-end Pipecat voice-pipeline demo for the megakernel-backed Qwen3-TTS.

Pipeline (the order required by the brief, Step 3 / Step 4)::

    mic | WAV file
      -> LocalAudioInputTransport  (or  WavFileInputProcessor)
        -> Silero VAD  (user-turn detection)
          -> DeepgramSTTService
            -> LLMUserAggregator
              -> Groq LLM (default; OpenAI / Anthropic also supported)
                -> MegakernelTTSService                  <-- our service
                  -> LocalAudioOutputTransport  (or AudioBufferProcessor -> WAV)
                    -> LLMAssistantAggregator

Why two input modes?

The GPU box (Ubuntu 22, NGC PyTorch 2.10) is typically headless -- no mic, no
speakers. To validate Step 4 (end-to-end) we support both:

* ``INPUT_MODE=mic`` (default on a workstation) -- uses
  ``LocalAudioInputTransport`` + ``LocalAudioOutputTransport``. Requires
  PyAudio + portaudio + a real audio device. Press Ctrl+C to exit.

* ``INPUT_MODE=file`` (recommended for first headless run) -- reads
  ``INPUT_WAV`` as 16 kHz mono int16, pushes ``InputAudioRawFrame``s through
  the pipeline, records the bot's TTS output to ``OUTPUT_WAV`` via
  ``AudioBufferProcessor``, then emits ``EndFrame`` once the file is
  exhausted. No mic, no speakers, no portaudio: pure file in / file out.

Env vars (loaded from ``.env`` next to this file):

    # --- STT ---
    DEEPGRAM_API_KEY=...

    # --- LLM (Groq is the brief's default) ---
    LLM_PROVIDER=groq                  # or "openai" or "anthropic"
    LLM_API_KEY=...
    LLM_MODEL=llama-3.1-8b-instant     # optional override

    # --- TTS ---
    MEGAKERNEL_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    MEGAKERNEL_MODEL_PATH=/workspace/qwen3-tts-1.7b
    MEGAKERNEL_SPEAKER=ryan
    MEGAKERNEL_DEVICE=cuda
    MEGAKERNEL_STUB=0                  # 1 -> silence stub for plumbing test
    HF_TOKEN=...                       # optional, for gated checkpoint pulls

    # --- Transport ---
    INPUT_MODE=mic                     # or "file"
    INPUT_WAV=/workspace/samples/user_utterance.wav  # required when INPUT_MODE=file
    OUTPUT_WAV=/workspace/samples/bot_response.wav   # used when INPUT_MODE=file

Run::

    cd inference-server
    cp .env.example .env  # fill in DEEPGRAM_API_KEY + LLM_API_KEY
    python3 pipecat_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import wave
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMRunFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.observers.user_bot_latency_observer import (
    LatencyBreakdown,
    UserBotLatencyObserver,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService

# Local-package import: pipecat_demo.py runs from inference-server/.
from megakernel_tts_service import MegakernelTTSService


class BotAudioRecorder(FrameProcessor):
    """Capture ONLY TTS output to a WAV. No user-input mixing.

    The brief's deliverable says the bot's TTS must be streamed out, so the
    saved file must be the bot's audio alone. Pipecat's stock
    AudioBufferProcessor merges user input + bot output into one buffer; that
    looked correct in earlier smoke tests but produced files dominated by the
    user utterance when the TTS stub returned silence, masking real
    regressions.
    """

    def __init__(self, output_path: str, sample_rate: int, num_channels: int) -> None:
        super().__init__()
        self._output_path = output_path
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._chunks: list[bytes] = []
        self._bot_speaking = False

    @property
    def total_bytes(self) -> int:
        return sum(len(c) for c in self._chunks)

    def save(self) -> None:
        audio = b"".join(self._chunks)
        _save_wav(self._output_path, audio, self._sample_rate, self._num_channels)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and frame.audio:
            # Synthesise BotStartedSpeakingFrame on the first TTS chunk in
            # headless file mode — normally LocalAudioTransport.output()
            # emits this when bot audio reaches the speakers, but we have no
            # output transport here. Without it, UserBotLatencyObserver can't
            # compute the canonical UserStopped→BotStarted e2e number.
            if not self._bot_speaking:
                self._bot_speaking = True
                await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
                await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
            self._chunks.append(frame.audio)
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        if self._bot_speaking:
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await super().cleanup()


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _require_env(key: str) -> str:
    """Read a required env var or exit with a clear message."""
    value = os.environ.get(key)
    if not value:
        logger.error(
            "Missing required env var {k}. Copy inference-server/.env.example "
            "to .env and fill it in.",
            k=key,
        )
        sys.exit(2)
    return value


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are a concise voice assistant. Your responses will be spoken aloud, "
    "so avoid emojis, lists, code blocks, or any formatting that can't be "
    "read by a TTS engine. Keep replies to one or two short sentences."
)


def _build_llm() -> object:
    """Return a Pipecat LLM service instance based on LLM_PROVIDER env var.

    Defaults to Groq (the brief's chosen provider). All three providers
    accept ``settings=<Service>.Settings(model=..., system_instruction=...)``;
    Groq inherits OpenAILLMSettings via its OpenAI-compatible base class.
    """
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    api_key = _require_env("LLM_API_KEY")

    if provider == "groq":
        from pipecat.services.groq.llm import GroqLLMService

        model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
        logger.info("LLM: Groq model={m}", m=model)
        service = GroqLLMService(
            api_key=api_key,
            settings=GroqLLMService.Settings(
                model=model,
                system_instruction=_SYSTEM_PROMPT,
            ),
        )
        # Warm the Groq HTTPS connection out of band so the first user turn
        # doesn't pay TLS handshake + DNS + cold connection pool RTT. Uses
        # the standalone groq SDK (not Pipecat) — Pipecat will set up its
        # own httpx pool later, but the OS-level TLS session cache + DNS
        # are now warm, shaving ~100-150 ms off the first turn's e2e.
        try:
            import time as _time
            from groq import Groq
            t0 = _time.perf_counter()
            Groq(api_key=api_key).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            logger.info(
                "Groq warmup OK in {ms:.0f} ms (DNS + TLS + cold-pool RTT paid here, not on first user turn)",
                ms=(_time.perf_counter() - t0) * 1000.0,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Groq warmup failed ({err!r}); first user turn will pay the cold cost.",
                err=e,
            )
        return service

    if provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService

        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        logger.info("LLM: OpenAI model={m}", m=model)
        return OpenAILLMService(
            api_key=api_key,
            settings=OpenAILLMService.Settings(
                model=model,
                system_instruction=_SYSTEM_PROMPT,
            ),
        )

    if provider == "anthropic":
        from pipecat.services.anthropic.llm import AnthropicLLMService

        model = os.environ.get("LLM_MODEL", "claude-3-5-haiku-latest")
        logger.info("LLM: Anthropic model={m}", m=model)
        return AnthropicLLMService(
            api_key=api_key,
            settings=AnthropicLLMService.Settings(
                model=model,
                system_instruction=_SYSTEM_PROMPT,
            ),
        )

    logger.error(
        "Unknown LLM_PROVIDER={p!r}. Use 'groq', 'openai', or 'anthropic'.",
        p=provider,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# File-based input (headless GPU mode)
# ---------------------------------------------------------------------------


class WavFileInputProcessor(FrameProcessor):
    """Stream a pre-recorded WAV through the pipeline as ``InputAudioRawFrame``s.

    The brief's STT (Deepgram streaming) wants 16 kHz mono int16. The WAV is
    chunked into 20 ms frames (matching ``LocalAudioInputTransport``'s
    ``num_frames = sample_rate / 100 * 2`` buffer size convention) and pushed
    downstream at real-time speed so the VAD + STT see a realistic stream.

    After the file is exhausted, an ``EndFrame`` is pushed downstream to
    cleanly terminate the pipeline.
    """

    def __init__(self, wav_path: str, *, realtime: bool = True) -> None:
        super().__init__()
        self._wav_path = wav_path
        self._realtime = realtime
        self._task = None

    async def cleanup(self) -> None:
        if self._task is not None:
            await self.cancel_task(self._task, timeout=2.0)
            self._task = None
        await super().cleanup()

    async def _pump(self) -> None:
        path = Path(self._wav_path)
        if not path.exists():
            logger.error("INPUT_WAV not found: {p}", p=path)
            await self.push_frame(EndFrame())
            return

        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()

            if sampwidth != 2:
                logger.error(
                    "INPUT_WAV must be 16-bit PCM (got sampwidth={sw})",
                    sw=sampwidth,
                )
                await self.push_frame(EndFrame())
                return

            # 20 ms chunks at the file's native rate. Deepgram accepts any
            # rate via the encoding/rate options; pipecat resamples
            # transparently downstream when needed.
            chunk_frames = max(1, sample_rate // 50)
            chunk_bytes = chunk_frames * num_channels * sampwidth
            frame_period_s = chunk_frames / sample_rate

            logger.info(
                "WavFileInputProcessor: pumping {p} ({sr} Hz, {ch} ch, "
                "{n} frames, {ms} ms chunks)",
                p=path,
                sr=sample_rate,
                ch=num_channels,
                n=wf.getnframes(),
                ms=int(frame_period_s * 1000),
            )

            # Synthesise a clean user-turn boundary so UserBotLatencyObserver
            # can fire in file mode. Without these the headless GPU run can't
            # measure the brief's e2e metric (VADUserStopped → BotStarted).
            await self.push_frame(VADUserStartedSpeakingFrame())
            await self.push_frame(UserStartedSpeakingFrame())

            while True:
                data = wf.readframes(chunk_frames)
                if not data:
                    break
                # Last chunk may be short; pad to full size so the VAD's
                # frame-size assertions hold.
                if len(data) < chunk_bytes:
                    data = data + b"\x00" * (chunk_bytes - len(data))

                await self.push_frame(
                    InputAudioRawFrame(
                        audio=data,
                        sample_rate=sample_rate,
                        num_channels=num_channels,
                    )
                )

                if self._realtime:
                    await asyncio.sleep(frame_period_s)

            # Mark end-of-user-turn for the latency observer. Pipecat fires
            # on_latency_measured between VADUserStopped and BotStarted, so
            # both halves of the boundary must be explicit in file mode.
            await self.push_frame(VADUserStoppedSpeakingFrame())
            await self.push_frame(UserStoppedSpeakingFrame())

        # Give the LLM + TTS time to react before tearing down. A production
        # build would wait on a TTSStoppedFrame; 12 s is enough for a Groq
        # roundtrip + a ~5 s spoken reply at our codec rate without dragging
        # the smoke-test wall-clock past a minute.
        drain_s = float(os.environ.get("FILE_MODE_DRAIN_S", "12.0"))
        logger.info("WavFileInputProcessor: input exhausted, draining for {d}s", d=drain_s)
        await asyncio.sleep(drain_s)
        await self.push_frame(EndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # We are a pure source. Forward everything (e.g. StartFrame from
        # the pipeline source) without modification.
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        # In pipecat 1.x there is no FrameProcessor.start() override hook
        # -- the framework dispatches StartFrame via a private __start. We
        # piggy-back on process_frame: the moment the StartFrame reaches us
        # (downstream direction), we kick off the WAV pump task.
        if isinstance(frame, StartFrame) and self._task is None:
            self._task = self.create_task(self._pump(), name="wav-input-pump")


def _build_latency_observer(metrics_sink: list[dict]) -> UserBotLatencyObserver:
    """Build a UserBotLatencyObserver wired to log + append per-cycle metrics.

    Pipecat's built-in observer fires three events:
    - ``on_latency_measured(observer, latency_seconds)`` once per user→bot
      cycle, measured from ``VADUserStoppedSpeakingFrame`` to
      ``BotStartedSpeakingFrame``. This is THE industry-standard voice-agent
      end-to-end metric per the brief's Step 4 "log t0 at UserStoppedSpeaking,
      t1 at BotStartedSpeaking" definition.
    - ``on_latency_breakdown(observer, breakdown)`` with a ``LatencyBreakdown``
      containing per-service TTFB metrics (Deepgram STT TTFB, LLM TTFB, our
      ``MegakernelTTSService`` TTFB) when ``enable_metrics=True``.
    - ``on_first_bot_speech_latency(observer, latency_seconds)`` once for the
      greeting / first audio (client-connect → first bot speech).

    Each event appends a dict to ``metrics_sink`` so the caller can dump JSON
    after the pipeline shuts down, alongside ``bench_results.json``.
    """
    observer = UserBotLatencyObserver()

    @observer.event_handler("on_latency_measured")
    async def _on_latency(_obs, latency_seconds: float):
        metrics_sink.append({"kind": "user_bot_latency_s", "value": latency_seconds})
        logger.info(
            "[E2E] UserStoppedSpeaking → BotStartedSpeaking: {ms:.0f} ms",
            ms=latency_seconds * 1000.0,
        )

    @observer.event_handler("on_latency_breakdown")
    async def _on_breakdown(_obs, breakdown: LatencyBreakdown):
        ttfbs = [
            {"processor": t.processor, "model": t.model, "ms": t.duration_secs * 1000.0}
            for t in breakdown.ttfb
        ]
        metrics_sink.append({
            "kind": "latency_breakdown",
            "user_turn_secs": breakdown.user_turn_secs,
            "ttfb": ttfbs,
        })
        for t in ttfbs:
            logger.info(
                "[E2E] per-service TTFB: {p}{model} = {ms:.0f} ms",
                p=t["processor"], model=f" ({t['model']})" if t["model"] else "",
                ms=t["ms"],
            )

    @observer.event_handler("on_first_bot_speech_latency")
    async def _on_first(_obs, latency_seconds: float):
        metrics_sink.append({"kind": "first_bot_speech_s", "value": latency_seconds})
        logger.info("[E2E] first bot speech: {ms:.0f} ms", ms=latency_seconds * 1000.0)

    return observer


def _dump_metrics(metrics_sink: list[dict], out_path: str | None) -> None:
    """Print the latency observer's metrics + optionally write to JSON.

    The observer events fire as the pipeline runs; this just summarises at
    teardown so the file-mode smoke test produces a single artifact the
    reviewer can diff against the brief's targets.
    """
    if not metrics_sink:
        logger.warning(
            "[E2E] no latency metrics captured — was VADUserStoppedSpeakingFrame "
            "ever emitted? Mic mode + a VAD analyzer + an actual user turn are required."
        )
        return
    e2e_values = [m["value"] for m in metrics_sink if m.get("kind") == "user_bot_latency_s"]
    if e2e_values:
        logger.info(
            "[E2E SUMMARY] user-bot latency n={n} mean={mean:.0f} ms min={mn:.0f} ms max={mx:.0f} ms",
            n=len(e2e_values),
            mean=sum(e2e_values) / len(e2e_values) * 1000.0,
            mn=min(e2e_values) * 1000.0,
            mx=max(e2e_values) * 1000.0,
        )
    if out_path:
        import json
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics_sink, f, indent=2)
        logger.info("[E2E] wrote {n} metric events to {p}", n=len(metrics_sink), p=out_path)


def _save_wav(path: str, audio: bytes, sample_rate: int, num_channels: int) -> None:
    """Write a 16-bit PCM WAV to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
    logger.info(
        "Wrote {b} bytes of {sr} Hz audio to {p}",
        b=len(audio),
        sr=sample_rate,
        p=path,
    )


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


async def _run_mic_mode(stt, llm, tts, context, aggregators) -> None:
    """Mic-in / speaker-out pipeline (workstation use)."""
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16_000,   # Deepgram-friendly
            audio_out_sample_rate=24_000,  # Qwen3-TTS native -> no resample
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    user_aggregator, assistant_aggregator = aggregators
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    metrics_sink: list[dict] = []
    latency_observer = _build_latency_observer(metrics_sink)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,
        ),
        observers=[latency_observer],
    )

    # Kick off with a greeting so the user knows the bot is alive.
    context.add_message(
        {"role": "developer", "content": "Greet the user in one short sentence."}
    )
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        _dump_metrics(metrics_sink, os.environ.get("METRICS_OUT"))


async def _run_file_mode(stt, llm, tts, context, aggregators) -> None:
    """File-in / WAV-out pipeline (headless GPU use)."""
    input_wav = _require_env("INPUT_WAV")
    output_wav = os.environ.get("OUTPUT_WAV", "/workspace/samples/bot_response.wav")

    wav_in = WavFileInputProcessor(input_wav, realtime=True)
    # Capture ONLY the bot's TTS frames so the saved WAV honestly reflects
    # what the megakernel produced — not the user's input mixed in.
    bot_recorder = BotAudioRecorder(
        output_path=output_wav,
        sample_rate=24_000,
        num_channels=1,
    )

    user_aggregator, assistant_aggregator = aggregators
    pipeline = Pipeline(
        [
            wav_in,
            stt,
            user_aggregator,
            llm,
            tts,
            bot_recorder,
            assistant_aggregator,
        ]
    )

    metrics_sink: list[dict] = []
    latency_observer = _build_latency_observer(metrics_sink)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=False,  # No user barge-in for file mode.
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=24_000,
        ),
        # Pipecat's default idle_timeout is 5 min — far too long for a
        # one-shot file run. The WAV pump pushes EndFrame after drain_s, but
        # the LLM+TTS path is the timeout-detection signal. 30s covers
        # Groq+TTS for a short reply with margin.
        idle_timeout_secs=float(os.environ.get("FILE_MODE_IDLE_TIMEOUT_S", "30.0")),
        observers=[latency_observer],
    )

    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        _dump_metrics(metrics_sink, os.environ.get("METRICS_OUT"))

    if bot_recorder.total_bytes == 0:
        logger.warning(
            "BotAudioRecorder captured 0 bytes — no TTSAudioRawFrame reached the recorder. "
            "Did the LLM emit a reply? Is the TTS stubbed and producing empty frames?"
        )
    bot_recorder.save()


async def run() -> None:
    """Build and run the voice pipeline."""
    # Do NOT override shell env: callers (e.g. agents flipping MEGAKERNEL_STUB=1
    # to exercise plumbing without a GPU) must be able to win over the
    # committed .env defaults.
    load_dotenv(override=False)
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))

    input_mode = os.environ.get("INPUT_MODE", "mic").lower()
    if input_mode not in {"mic", "file"}:
        logger.error("INPUT_MODE must be 'mic' or 'file', got {m!r}", m=input_mode)
        sys.exit(2)

    # MEGAKERNEL_STUB accepts:
    #   "0" (default) -> real megakernel Qwen3-TTS — the brief's deliverable
    #   "1"           -> silence frames (plumbing-only — does NOT meet brief)
    raw_stub = os.environ.get("MEGAKERNEL_STUB", "0").lower()
    stub = raw_stub in {"1", "silence"}
    speaker = os.environ.get("MEGAKERNEL_SPEAKER", "ryan")

    # ---- Services -------------------------------------------------------
    stt = DeepgramSTTService(api_key=_require_env("DEEPGRAM_API_KEY"))
    llm = _build_llm()

    try:
        tts = MegakernelTTSService(
            model_name=os.environ.get(
                "MEGAKERNEL_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            ),
            model_path=os.environ.get(
                "MEGAKERNEL_MODEL_PATH", "/workspace/qwen3-tts-1.7b"
            ),
            speaker=speaker,
            device=os.environ.get("MEGAKERNEL_DEVICE", "cuda"),
            stub=stub,
        )
    except Exception as exc:  # noqa: BLE001 - surface ALL init failures clearly
        logger.exception(
            "MegakernelTTSService failed to initialize. Set MEGAKERNEL_STUB=1 "
            "to bypass the model and exercise the Pipecat plumbing only."
        )
        raise SystemExit(3) from exc

    # ---- Context + aggregators -----------------------------------------
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        # Pass the VAD analyzer so the user aggregator can drive the default
        # VADUserTurnStartStrategy / TranscriptionUserTurnStartStrategy pair.
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    logger.info(
        "INPUT_MODE={m} stub={s} speaker={sp}",
        m=input_mode, s=stub, sp=speaker,
    )

    if input_mode == "mic":
        await _run_mic_mode(stt, llm, tts, context, aggregators)
    else:
        await _run_file_mode(stt, llm, tts, context, aggregators)


def main() -> int:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("interrupted, exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
