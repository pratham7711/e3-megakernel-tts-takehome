# inference-server

Wires AlpinDale's `qwen_megakernel` (CUDA megakernel for Qwen3 decode on RTX
5090) as the decode backend for Qwen3-TTS's Talker, then streams audio chunks
into a Pipecat voice pipeline.

## Layout

| File                          | Role                                                          |
| ----------------------------- | ------------------------------------------------------------- |
| `megakernel_tts.py`           | Importable Qwen3-TTS pipeline wrapper (Talker + code pred. + codec). Async `generate()` yields int16 PCM @ 24 kHz. |
| `megakernel_tts_service.py`   | Pipecat `TTSService` subclass; wraps `MegakernelTTS`.         |
| `bench_megakernel.py`         | Honest bench: decode tok/s, TTFC, RTF -> `bench_results.json`.|
| `pipecat_demo.py`             | Runnable demo: mic -> Deepgram -> LLM -> our TTS -> speaker.  |
| `.env.example`                | Template env vars for the demo.                               |
| `requirements.txt`            | Python deps. `qwen_megakernel` installed separately.          |

## Install (remote RTX 5090 box)

```bash
# 1. Megakernel (editable, built against local CUDA 12.x)
pip install -e ~/qwen_megakernel/

# 2. Everything else
pip install -r requirements.txt

# 3. Env
cp .env.example .env
$EDITOR .env  # fill in DEEPGRAM_API_KEY, LLM_API_KEY
```

## Run the benchmark

The bench measures three of the four take-home metrics; end-to-end latency
is measured by the Pipecat demo and is intentionally out of scope here so we
don't conflate TTS perf with STT/LLM perf.

```bash
# Full bench: decode tok/s + TTFC + RTF, 3 warmup + 5 timed each.
python bench_megakernel.py

# Plumbing check without GPU (uses silence stub).
python bench_megakernel.py --stub --skip-decode

# Just TTFC + RTF, more iterations.
python bench_megakernel.py --skip-decode --warmup 5 --timed 20
```

Output:
- pretty-printed table to stdout with `mean +/- stdev` and `(min, max, n)`
- machine-readable JSON at `bench_results.json` with every per-run sample

Target perf (from the brief, tightest set):
- Decode tok/s: as high as the kernel goes (Qwen3-0.6B baseline ~ 1k tok/s on
  5090, expected ~600 tok/s after the 1.7B port)
- TTFC: < 50 ms
- RTF: < 0.10

## Run the Pipecat demo

```bash
python pipecat_demo.py
```

Talks via local mic / speakers. Press Ctrl-C to stop. Requires PyAudio +
working system audio.

### Plumbing-only smoke test (no GPU)

Set `MEGAKERNEL_STUB=1` in `.env`. The TTS service will emit silence sized
roughly to the input text, so you can verify Deepgram + LLM are wired
correctly before the kernel mods land.

## Architecture sketch

```
text (from LLM)
  |
  v
Talker (megakernel CUDA decode, ~13 Hz, vocab 3072)
  |    one semantic codec token per step
  v
Code Predictor (5 layers, 16 codebook heads, vocab 2048)
  |    one 16-tuple of codebook ids per Talker step
  v
Codec Decoder (Qwen3-TTS-Tokenizer-12Hz, non-DiT)
  |    ~1920 fp32 samples per codec frame
  v
int16 LE PCM bytes, yielded async at 12.5 frames/sec @ 24 kHz
  |
  v
Pipecat TTSAudioRawFrame(sample_rate=24000, num_channels=1)
```

The **Talker** is the only stage running through the megakernel; the code
predictor and codec decoder are vanilla PyTorch on the same GPU. This matches
the brief: swap the megakernel in for decode, don't reinvent the whole stack.

## Sample-rate decision

Qwen3-TTS native rate is **24 kHz**. We expose audio at 24 kHz throughout the
service; the `LocalAudioOutputTransport` is also configured at 24 kHz so
there is no resampling on the hot path. Pipecat will resample only if a
downstream pipeline asks for a non-24-kHz `audio_out_sample_rate` at
`StartFrame` time (handled by the inherited stream resampler).

## Audio buffering

We yield exactly one codec frame per `TTSAudioRawFrame` (~80 ms of audio at
12.5 Hz codec rate, ~3.8 KB at 24 kHz int16). Pipecat's output transport
handles smoothing / jitter buffering downstream; we do not pre-batch.

## Gotchas

- **megakernel is RTX 5090 only.** It targets sm_120; do not try this on a
  3090 or A100, the CUDA build will refuse.
- **kernel mods pending.** Until then, calling `MegakernelTTS.generate()`
  without `stub=True` raises `NotImplementedError`. The Pipecat service
  accepts a `stub=True` kwarg that gets you a silent-but-runnable pipeline.
- **Deepgram needs an internet connection.** The "local-only" part of this
  stack is just the TTS. Swap in a local STT (e.g. `whisper.cpp`) if that
  matters.
- **PyAudio + macOS.** `LocalAudioTransport` needs `pyaudio`, which on macOS
  needs `brew install portaudio` first. The remote box is Linux, so this
  matters only for local dev.

## Open questions

- Code predictor invocation cadence -- does it need a context of `N>1`
  Talker tokens before emitting codebook ids, or is `N=1` streaming OK?
  Determines TTFC floor. Verify on remote.
- Speaker embedding format -- the `CustomVoice` repo ships a speaker map but
  the exact tensor key is undocumented in the README; check the modeling
  code path on the remote.
- Whether codec output is fp32 vs. bf16 -- affects the quantization step in
  `_f32_to_i16_bytes`. Either is fine; we just need to pick one.
