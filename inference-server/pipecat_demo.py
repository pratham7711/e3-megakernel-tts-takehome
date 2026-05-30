"""Runnable Pipecat voice-pipeline demo for the megakernel-backed Qwen3-TTS.

Pipeline:

    mic -> LocalAudioInputTransport
        -> Silero VAD (user turn detection)
        -> DeepgramSTTService
        -> LLMContextAggregator (user)
        -> LLM (Anthropic claude-3-5-haiku OR OpenAI gpt-4o-mini, env-switchable)
        -> MegakernelTTSService                          <-- our service
        -> LocalAudioOutputTransport
        -> LLMContextAggregator (assistant)

Why ``LocalAudioOutputTransport``: the brief asks for an end-to-end demo on a
single box (RTX 5090 + nearby mic/speakers) without a browser or WebRTC stack.
PyAudio + ALSA / CoreAudio is the simplest path to "press play, talk to it".
Swap in ``DailyTransport`` or ``SmallWebRTCTransport`` later if remote demo
becomes a requirement.

Env vars (loaded from ``.env``):
    DEEPGRAM_API_KEY=...
    LLM_PROVIDER=anthropic   # or "openai" or "groq"
    LLM_API_KEY=...          # used by whichever provider is selected
    LLM_MODEL=...            # optional override
    MEGAKERNEL_SPEAKER=ryan  # optional override
    MEGAKERNEL_MODEL_PATH=/workspace/qwen3-tts-1.7b  # optional local path
    MEGAKERNEL_STUB=0        # set to 1 for plumbing smoke test (silence)

Run::

    python pipecat_demo.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

# Local-package import; this file is run from ``inference-server/``.
from megakernel_tts_service import MegakernelTTSService


def _build_llm() -> object:
    """Return a Pipecat LLM service instance based on LLM_PROVIDER env var."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    api_key = os.environ["LLM_API_KEY"]

    system_prompt = (
        "You are a concise voice assistant. Your responses will be spoken "
        "aloud, so avoid emojis, lists, code blocks, or any formatting that "
        "can't be read by a TTS engine. Keep replies to one or two short "
        "sentences."
    )

    if provider == "anthropic":
        from pipecat.services.anthropic.llm import AnthropicLLMService

        model = os.environ.get("LLM_MODEL", "claude-3-5-haiku-latest")
        return AnthropicLLMService(
            api_key=api_key,
            settings=AnthropicLLMService.Settings(
                model=model,
                system_instruction=system_prompt,
            ),
        )

    if provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService

        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        return OpenAILLMService(
            api_key=api_key,
            settings=OpenAILLMService.Settings(
                model=model,
                system_instruction=system_prompt,
            ),
        )

    if provider == "groq":
        # GroqLLMService is an OpenAI-compatible subclass shipped at
        # pipecat.services.groq.llm.GroqLLMService; it inherits the
        # ``Settings`` dataclass from BaseOpenAILLMService (model +
        # system_instruction supported).
        from pipecat.services.groq.llm import GroqLLMService

        model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
        return GroqLLMService(
            api_key=api_key,
            settings=GroqLLMService.Settings(
                model=model,
                system_instruction=system_prompt,
            ),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'anthropic', 'openai', or 'groq'."
    )


async def run() -> None:
    """Build and run the voice pipeline."""
    load_dotenv(override=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    stub = os.environ.get("MEGAKERNEL_STUB", "0") == "1"
    speaker = os.environ.get("MEGAKERNEL_SPEAKER", "ryan")

    # ---- Transport: local mic in, local speaker out ----------------------
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16_000,   # Deepgram likes 16 kHz; cheap to capture
            audio_out_sample_rate=24_000,  # match Qwen3-TTS native -> no resample
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # ---- Services --------------------------------------------------------
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = _build_llm()
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

    # ---- Context aggregation --------------------------------------------
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # ---- Pipeline -------------------------------------------------------
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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,
        ),
    )

    # Kick off with a greeting so the user knows the bot is alive.
    context.add_message(
        {"role": "developer", "content": "Greet the user in one short sentence."}
    )
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner()
    await runner.run(task)


def main() -> int:
    import asyncio

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("interrupted, exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
