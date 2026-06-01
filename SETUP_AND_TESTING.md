# Setup & Testing — Reproducibility Walkthrough

> Companion to the entry-point [`README.md`](./README.md). README is the 3-minute overview. **This document is the full step-by-step setup + testing guide for a reviewer who wants to reproduce the numbers from scratch.**
>
> Cost: a full reproduction takes ~10-15 minutes of wallclock and ~$0.30 of Vast.ai compute. The most expensive step is downloading the Qwen3-TTS weights (~3.5 GB safetensors), bottlenecked by network.

---

## 0. Prerequisites — what you need before starting

| | Required | Verify with |
|---|---|---|
| GPU | RTX 5090 (Blackwell, sm_120) — the kernel is hand-tuned for this | `nvidia-smi --query-gpu=name --format=csv,noheader` |
| CUDA | ≥ 12.8 (PyTorch NGC nightly bundle is easiest) | `nvcc --version` |
| Python | 3.11 or 3.12 | `python3 --version` |
| PyTorch | 2.10.0a or newer (NGC image ships this) | `python3 -c "import torch; print(torch.__version__)"` |
| Disk | ≥ 40 GB free | `df -h /workspace` |
| RAM | ≥ 32 GB | `free -g` |
| Internet | Outbound HTTPS for Deepgram + Groq APIs | `curl -s https://api.deepgram.com/v1/projects -I` |
| API keys | Deepgram + one LLM provider (Groq is what we use) | see step 4 |

**Local-machine requirements** (the one you'll SSH from):
- SSH client (default `ssh` works)
- A web browser (Chrome recommended for Gradio UI)
- `curl` and `jq` (optional, but helpful for inspecting results)

---

## 1. Rent a GPU (~2 min)

On [vast.ai](https://cloud.vast.ai/create/):

1. Filter for `RTX 5090`, CUDA `≥ 12.8`, RAM `≥ 32 GB`, Disk `≥ 40 GB`.
2. Choose the cheapest interruptible. Hourly cost is typically $0.80-1.10.
3. In the instance template, select the **NVIDIA PyTorch NGC image** (e.g. `nvcr.io/nvidia/pytorch:25.01-py3`). This bundles PyTorch 2.10.0a + CUDA 12.8 + cuDNN.
4. Hit Rent. Wait for status → `running` (~30 seconds).
5. SSH to it:
   ```bash
   ssh -p <port> root@<host>
   ```
   Host + port are on the Vast instance dashboard. Add to your `~/.ssh/config` as `e3-vast` if you want shorter commands later.

**Cost-control habit**: stop the instance when you're done. Even idle instances bill by the hour.

---

## 2. Clone the repo (~30 sec)

```bash
cd /workspace
git clone https://github.com/pratham7711/e3-megakernel-tts-takehome.git e3-megakernel-tts
cd e3-megakernel-tts
```

Verify you got the right commit (latest master should be `9a3bc8a` or newer at the time of writing):
```bash
git log --oneline -3
```

---

## 3. Python environment (~2 min)

The NGC PyTorch image ships PyTorch already. **Do NOT pin or downgrade torch** — the megakernel JIT-compiles against the version it finds at import time, and downgrading breaks the kernel ABI.

```bash
pip install --break-system-packages safetensors transformers triton ninja accelerate
pip install --break-system-packages -r inference-server/requirements.txt
```

Then install the megakernel as an editable package — this triggers the first `nvcc` build of `kernel.cu` (takes ~60-90s on cold cache, ~8s on warm):

```bash
pip install --break-system-packages -e qwen_megakernel_modified/
```

Verify the kernel built:
```bash
python3 -c "
import torch
from qwen_megakernel.build import get_extension
get_extension()
print('decode op:       ', hasattr(torch.ops.qwen_megakernel_C, 'decode'))
print('decode_embed op: ', hasattr(torch.ops.qwen_megakernel_C, 'decode_embed'))
"
```

Expected output:
```
decode op:        True
decode_embed op:  True
```

Both must be `True`. The `decode_embed` op is the precomputed-embedding entry added by this submission (see [`CHANGELOG.md`](./CHANGELOG.md)).

---

## 4. API keys (~1 min)

```bash
cp inference-server/.env.example inference-server/.env
$EDITOR inference-server/.env
```

Fill in:
- `DEEPGRAM_API_KEY` — for STT + audio QA. Sign up at [console.deepgram.com](https://console.deepgram.com/) (free tier has enough credits for many bench runs).
- `LLM_API_KEY` — for the voice agent. We use Groq (free tier, fastest TTFB). Sign up at [console.groq.com](https://console.groq.com/).

Verify:
```bash
grep -c "=." inference-server/.env  # should be ≥ 2 (both keys set)
```

---

## 5. Download Qwen3-TTS weights (~5 min, network-bound)

```bash
mkdir -p /workspace/qwen3-tts-1.7b
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
    local_dir='/workspace/qwen3-tts-1.7b',
)
print('OK')
"
```

(If you've never logged into HuggingFace on this box, this may prompt for `huggingface-cli login`. The Qwen3-TTS model is publicly accessible — no special permissions needed.)

Verify:
```bash
ls -la /workspace/qwen3-tts-1.7b/model.safetensors
# expected: ~3.5 GB
```

---

## 6. Smoke test — first end-to-end synthesis (~30s)

This verifies the full pipeline (megakernel + CodePredictor + codec + sampling) works on a single short utterance:

```bash
cd inference-server
PYTHONPATH=/workspace/qwen_megakernel_modified python3 -c "
import asyncio, time
from megakernel_tts import MegakernelTTS

tts = MegakernelTTS(
    model_name='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
    model_path='/workspace/qwen3-tts-1.7b',
    speaker='ryan',
    device='cuda',
)

async def go():
    pcm = bytearray()
    t0 = time.perf_counter()
    first = None
    async for chunk in tts.generate('Hello, this is a test of the megakernel speech synthesis system.'):
        if first is None:
            first = (time.perf_counter() - t0) * 1000
        pcm.extend(chunk)
    wall = (time.perf_counter() - t0) * 1000
    dur = (len(pcm) // 2) / 24000
    print(f'TTFC={first:.1f}ms wall={wall:.0f}ms audio={dur:.2f}s RTF={(wall/1000)/dur:.3f} bytes={len(pcm)}')

asyncio.run(go())
"
```

Expected:
- Load time ~13 sec (cold)
- After load: `TTFC=25-30 ms wall=500-800ms audio=4-5s RTF=0.14-0.16`
- No exceptions, no `RuntimeError: decode_embed op not registered`, no Deepgram errors

If `decode_embed op not registered` shows up: rebuild the kernel:
```bash
rm -rf ~/.cache/torch_extensions/*/qwen_megakernel_C
# then re-import in a fresh Python process
```

---

## 7. Run the canonical bench (~3 min)

This is the bench whose output sits in `bench_results.json` and is cited everywhere in the README, CHANGELOG, and email.

```bash
cd inference-server
PYTHONPATH=/workspace/qwen_megakernel_modified python3 bench_megakernel.py --warmup 3 --timed 5 --skip-decode
```

Expected output (the bottom block):
```
================================================================
 Megakernel Qwen3-TTS bench results
================================================================
 model         : Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
 speaker       : ryan
 device        : cuda
 ...
 TTFC               : 25.3 +/- 0.0 ms  (min=25.3, max=25.4, n=5)
 RTF                : 0.1452 +/- 0.0000  (min=0.1450, max=0.1454, n=5)
================================================================
```

Pretty-printed tier table:
```bash
python3 ../scripts/show_bench.py
```

Why `--skip-decode`: the tok/s sub-bench loads a second `Decoder` instance which conflicts with the served instance's kernel static barriers. The TTFC + RTF bench above is the canonical one. See [`BENCHMARK_REPORT.md`](./BENCHMARK_REPORT.md) §"Honest gaps in measurement".

---

## 8. Audio QA gate (~30s)

Generate a test utterance:
```bash
PYTHONPATH=/workspace/qwen_megakernel_modified python3 -c "
import asyncio, numpy as np, soundfile as sf
from megakernel_tts import MegakernelTTS
tts = MegakernelTTS(model_path='/workspace/qwen3-tts-1.7b', speaker='ryan', device='cuda')
async def go():
    pcm = bytearray()
    async for chunk in tts.generate('Hello. How are you doing today?', max_new_tokens=200):
        pcm.extend(chunk)
    sf.write('/tmp/qa_check.wav', np.frombuffer(bytes(pcm), dtype=np.int16), 24000)
asyncio.run(go())
"
```

Round-trip through Deepgram:
```bash
bash ../scripts/deepgram_stt_check.sh /tmp/qa_check.wav
```

Expected:
```
==================== DEEPGRAM RESULT ====================
Transcript: Hello. How are you doing today?
Confidence: 1.0
=========================================================
```

**Pass criterion**: confidence ≥ 0.95. We typically see 1.000.

To cross-check against vanilla upstream Qwen3-TTS (the control we benchmark against):
```bash
python3 ../scripts/upstream_ref_test.py
bash ../scripts/deepgram_stt_check.sh ../samples/upstream_ref_ryan.wav
# expected: same transcript, confidence ~0.9995
```

---

## 9. Run the Pipecat demo (the brief's e2e gate)

### Mic mode — interactive voice agent

```bash
cd inference-server
PYTHONPATH=/workspace/qwen_megakernel_modified python3 pipecat_demo.py INPUT_MODE=mic OUTPUT_MODE=local
```

This wires `DeepgramSTTService → GroqLLMService → MegakernelTTSService → LocalAudioOutputTransport`. Speak into the GPU box's audio input (if it has one — Vast.ai typically doesn't; use the UI mode below from your laptop instead). Ctrl-C to exit.

### File mode — deterministic e2e via WAV input

```bash
bash ../scripts/run_voice_turn.sh ../samples/user_utterance.wav /tmp/bot_reply.wav
# bot_reply.wav contains the synthesized reply
# play locally:
afplay /tmp/bot_reply.wav  # macOS
```

This is the failsafe path used in the demo recording if the browser UI hits any glitch.

### UI mode — Gradio interface for browser-based testing

On the GPU:
```bash
cd inference-server
PYTHONPATH=/workspace/qwen_megakernel_modified python3 ui_v2.py
# Listens on 0.0.0.0:8080. Takes ~10 sec to warm up.
```

On your local laptop:
```bash
ssh -f -N -L 8080:localhost:8080 root@<vast-host>
open http://localhost:8080  # macOS; Linux: xdg-open
```

In Chrome, you'll see:
- The voice-agent UI on the left (record mic → Send → bot replies)
- Live metric cards showing the **canonical bench numbers** (25.3 ms / 0.1452)
- A subtitle below the cards showing the **per-turn live measurement** (typically 60-90 ms TTFC inside Gradio's HTTP-server async context — see [`BENCHMARK_REPORT.md`](./BENCHMARK_REPORT.md) §"UI vs bench measurement")

---

## 10. Verify against upstream Qwen3-TTS

We cross-validate against vanilla upstream by running the same input through `transformers`'s `Qwen3TTSForConditionalGeneration` and Deepgram-checking the output:

```bash
PYTHONPATH=/workspace/qwen_megakernel_modified python3 ../scripts/upstream_ref_test.py \
    --text "Hello. How are you doing today?" \
    --speaker ryan \
    --out ../samples/upstream_ref_ryan.wav
bash ../scripts/deepgram_stt_check.sh ../samples/upstream_ref_ryan.wav
```

Expected: identical Deepgram transcript, confidence ~0.9995. **Our megakernel-wired output matches upstream within 0.0005 Deepgram delta.** This is the "wiring is faithful" claim from the README.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: decode_embed op not registered` | Stale torch extension cache | `rm -rf ~/.cache/torch_extensions/*/qwen_megakernel_C`, re-import |
| `FileNotFoundError: model.safetensors` in `qwen3-tts-1.7b` | Weights not downloaded | Run §5 again |
| `Address already in use` on Gradio port 8080 | Old UI process still running | `pkill -9 -f ui_v2.py; sleep 2; restart` |
| Deepgram returns 0.0 confidence | API key wrong OR audio is actually broken | Verify `DEEPGRAM_API_KEY` is set in `.env`; if API key is fine, the audio path has a regression — bisect via `git log --oneline` |
| `RuntimeError: Cannot run the event loop while another loop is running` during warmup | UI's `_warmup_full_ar` collision with Gradio's loop | Benign — warmup falls back, first user turn pays the cold-compile cost (~50ms extra TTFC). Subsequent turns are fast. |
| Bench numbers different from canonical | Different sampling temperature OR not running megakernel-AR path | Check `QWEN_USE_MEGAKERNEL_AR` env var — should be `1` or unset (default ON). Verify in the bench log: `grep -E "step_embed_megakernel\|decode_embed" /tmp/bench.log` |
| Two-`Decoder` deadlock in `bench_decode_tok_per_s` sub-bench | Kernel static barriers shared across instances | Use `--skip-decode` flag (documented in [`BENCHMARK_REPORT.md`](./BENCHMARK_REPORT.md)) |

---

## 12. Cleanup — when you're done

```bash
# On the GPU box:
pkill -9 -f ui_v2.py
pkill -9 -f pipecat_demo

# On your laptop:
kill $(lsof -ti:8080) 2>/dev/null  # drop the SSH tunnel
```

**Then stop + destroy the Vast instance from the dashboard** (or via the Vast API). Vast bills by the hour even when idle.

---

## See also

- [`README.md`](./README.md) — entry point (architecture, kernel mods, how to run)
- [`BENCHMARK_REPORT.md`](./BENCHMARK_REPORT.md) — honest measurement report with full methodology
- [`ENGINEERING_NOTES.md`](./ENGINEERING_NOTES.md) — process + tradeoffs + why we miss Tightest tier
- [`CHANGELOG.md`](./CHANGELOG.md) — chronological diff with bench numbers
- [`DEMO_SCRIPT.md`](./DEMO_SCRIPT.md) — recording script for the demo video
