# inference-server

The TTS pipeline + Pipecat wiring + bench harness + Gradio UI.

> Install + how-to-run lives in the [top-level README](../README.md). This file is a folder-local map.

## Layout

| File | Role |
|---|---|
| `megakernel_tts.py` | Async streaming wrapper. `generate(text)` yields raw int16 PCM bytes per codec frame. Drives the megakernel + CodePredictor + codec. |
| `megakernel_tts_service.py` | Pipecat `TTSService` subclass. Yields `TTSAudioRawFrame` per chunk for the Pipecat path; also exposes a public `stream_tts()` returning raw bytes for non-Pipecat callers. |
| `bench_megakernel.py` | Honest bench: TTFC + RTF + decode tok/s. Explicit `cuda.synchronize()` at every timer boundary. Writes `bench_results.json`. |
| `pipecat_demo.py` | Runnable e2e voice agent: mic → Deepgram STT → Groq LLM → MegakernelTTSService → speaker. File-mode + mic-mode. |
| `ui_v2.py` | Gradio UI v2 — record-and-send mic widget, live metric cards reading canonical bench, per-stage timing log, "honest disclosures" right rail. |
| `qwen3_tts_components.py` | CodePredictor + codec PyTorch modules + weight loaders. The 15-step AR sub-decode lives here (`CodePredictor.generate_ar`). |
| `.env.example` | Template — `DEEPGRAM_API_KEY` + `LLM_API_KEY` (Groq). |
| `requirements.txt` | Python deps. The megakernel itself is installed separately (`pip install -e ../qwen_megakernel_modified/`). |

## Quick commands

```bash
# Bench (5 timed + 3 warmup):
python3 bench_megakernel.py --warmup 3 --timed 5
python3 ../scripts/show_bench.py   # pretty-print tier table

# Pipecat demo, mic mode:
python3 pipecat_demo.py INPUT_MODE=mic OUTPUT_MODE=local

# Pipecat demo, file mode (no live mic; deterministic):
bash ../scripts/run_voice_turn.sh ../samples/user_utterance.wav /tmp/bot.wav
afplay /tmp/bot.wav

# Gradio UI (browse via ssh tunnel from laptop):
PYTHONPATH=../qwen_megakernel_modified python3 ui_v2.py
```

## Sample rate

24 kHz native (Qwen3-TTS codec rate). `MegakernelTTSService` resamples to whatever the Pipecat pipeline asks for; default for `LocalAudioOutputTransport` is also 24 kHz, so resampling is a no-op.

## Streaming guarantee

`MegakernelTTS.generate()` is an `AsyncGenerator` yielding one ~80 ms PCM chunk per codec frame (12.5 Hz). The first chunk is yielded as soon as the first frame is decoded — no end-of-utterance buffering. That's the brief's "stream chunks, do NOT buffer" requirement. Verified at three layers:

1. Direct: `async for chunk in tts.generate(text)` in `bench_megakernel.py` — TTFC measured at first yield.
2. Pipecat path: `MegakernelTTSService.run_tts()` yields `TTSAudioRawFrame` per codec frame.
3. Public bytes: `MegakernelTTSService.stream_tts()` yields raw bytes for non-Pipecat consumers (Gradio UI).

The only consumer that breaks streaming end-to-end is the Gradio UI sink (collects + single end-of-utterance blob) — reason documented inline in `ui_v2.py` and in the UI's "Honest disclosures" panel.
