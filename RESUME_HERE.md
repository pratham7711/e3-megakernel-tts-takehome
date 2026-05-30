# Resume marker — e3 take-home (2026-05-31 03:50 IST)

> **Start here every new session.** Single source of truth for what's done, what's left, and how to continue.

---

## TL;DR — where we are

- **Repo**: https://github.com/pratham7711/e3-megakernel-tts-takehome (public). Local HEAD ahead of origin — needs `git push`.
- **Status**: ~98% done. **ALL BRIEF PERFORMANCE BENCHMARKS PASS AT TIGHTEST TIER.** Only remaining items: audio-intelligibility fix (separate from perf) and demo video.
- **Deadline**: 2026-06-01 EOD (~18 h from this marker)
- **Spend**: ~$1.20 of $2.30 GPU headroom used in tonight's iteration ($7.70 + $1.20 = $8.90 of $10 budget total).
- **GPU**: instance `#38639051` (Pennsylvania, $1.336/hr) is STOPPED. The old #38548758 was destroyed (host took its GPU back, queued indefinitely).

## 🎯 FINAL BENCHMARK NUMBERS (n=5, 3 warmup, RTX 5090 sm_120, explicit cuda.sync)

| Metric | Value | Tightest | Perf | Deliverables | Verdict |
|---|---|---|---|---|---|
| **TTFC** | **35.41 ± 0.08 ms** | <50 ✅ | <60 ✅ | <90 ✅ | **PASS ALL 3** |
| **RTF** | **0.0558 ± 0.0000** | <0.1 ✅ | <0.15 ✅ | <0.3 ✅ | **PASS ALL 3** |
| **Decode tok/s** (1.7B talker) | **429.7 ± 0.03** | — | — | — | report-only |
| **E2E** (UserStopped → BotStarted, Pipecat warm) | **916 ms** | — | — | — | Groq 652 ms cloud + megakernel 36 ms + pipeline 228 ms |

Cross-validation: Pipecat measures megakernel TTS TTFB at 36 ms; standalone bench measures TTFC at 35.4 ms. Same number — methodology consistent.

Raw data in `bench_results.json` + `metrics_gpu.json`. Logs in `bench_runs/`.

## The ONE engineering insight that did all the work

`torch.compile(mode="reduce-overhead")` was wired in but silently disabled by a CUDA-graph storage-reuse error. Root cause: RoPE tables were being built INSIDE `forward()` and assigned to module attributes (`self._cos_table = cos`), which placed them in the CUDA-graph private pool. Second compiled call → storage reused → RuntimeError.

Fix (commit `1c958e4`): hoist RoPE construction to `__init__` (pre-build for max seq len on CPU+fp32, then `warmup_rope(device, dtype)` to materialize on GPU eagerly BEFORE any compiled call). 25-line patch. RTF dropped 0.32 → 0.056 (6× speedup).

Plus init-time warmup in `MegakernelTTS.__init__` so Pipecat's first user turn doesn't pay the 22 s cold-compile cost.

---

## Decision the human needs to make first (tomorrow)

The benchmark numbers are LOCKED IN at the brief's tightest tier. Three open items remain:

**Item 1 — Audio intelligibility (NEW, surfaced 2026-05-31)**: the megakernel produces real broadband audio but the talker doesn't reliably emit EOS and the speaker "ryan" isn't being tokenized into the prompt. Result: speech-like babble, not intelligible English. Fix is documented (build upstream chat template with `<|audio_bos|>` + speaker control tokens) but is a ~2-3 h reverse-engineering job. Until this is fixed the demo recording is risky.

**Item 2 — Demo video**: brief requires it. Can be:
- (a) The full Pipecat mic→GPU→speaker loop with INTELLIGIBLE audio (needs Item 1 fixed)
- (b) The Mac-side `ui_loopback.py` Tab 2 voice loop with macOS `say` substitute — proves plumbing end-to-end, audio IS intelligible (because it's macOS speech), but it's the substitute path
- (c) Walk-through of `bench_megakernel.py` running + the metric table — pure perf demo, no voice agent at all

**Item 3 — Git push + email**: HEAD `1c958e4` is local-only. Need `git push` + send `~/brain/build/side-projects/e3-submission-email-draft.md`.

**Default plan for tomorrow**: fix Item 1 (audio intelligibility, ~2 h on the GPU), then record video (option a), then push + send. Budget ~$1.10 GPU headroom — tight but doable.

If Item 1 fix doesn't work in 2 h, fall back to (b) for the video.

---

## Tomorrow's actionable checklist

Run these from `~/Documents/Repositories/e3-megakernel-tts/`.

- [ ] **1. Restart GPU + redeploy** — instance id is now `38639051` (Pennsylvania, $1.336/hr, fresh tonight). To start:
  ```bash
  E3_INSTANCE_ID=38639051 bash scripts/deploy.sh
  ```
  The deploy script's `vastai` binary fix landed tonight — it now uses `/tmp/vastvenv/bin/vastai` because the system Python 3.9 `vastai` is broken by `match`-statement syntax.

  After deploy, you'll need to re-install Python deps + re-download Qwen3-TTS weights (the previous instance was destroyed). Helper sequence:
  ```bash
  ssh e3-vast 'apt-get install -y portaudio19-dev'
  ssh e3-vast 'pip install --no-deps -U "transformers==4.57.3" safetensors numpy scipy "huggingface-hub>=0.34,<1.0"'
  ssh e3-vast 'pip install -U "pipecat-ai[deepgram,groq,silero,openai,anthropic,local]>=1.3,<2.0" gradio soundfile loguru python-dotenv'
  HF_TOKEN=<from .env> ssh e3-vast 'python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\", local_dir=\"/workspace/qwen3-tts-1.7b\", token=\"$HF_TOKEN\")"'
  scp samples/user_utterance.wav e3-vast:/workspace/samples/
  ```

- [ ] **2. Verify in browser** — open http://localhost:8080. Click Generate with the default text. **Expect**: audio should sound DIFFERENT from the previous run (this session committed the **semantic codebook 4096 fix** — the prior runs produced garbled audio because the semantic codebook was at random init). If the new audio still sounds broken, see `~/brain/build/side-projects/e3-final-code-review-2.md` for the next 8 LOW-severity findings.

- [ ] **3. Run the live Pipecat e2e** — proves the brief's Step 3 and gives us the missing 4th metric (end-to-end latency). NOTE: `pipecat_demo.py` now parses `MEGAKERNEL_STUB` as `0|silence|mac_say` and accepts `FILE_MODE_DRAIN_S` / `FILE_MODE_IDLE_TIMEOUT_S` overrides — for the real GPU run leave `MEGAKERNEL_STUB` unset (default 0 = real megakernel):
  ```bash
  ssh e3-vast 'cd /workspace/inference-server && \
    SSL_CERT_FILE="$(python3 -c "import certifi; print(certifi.where())")" \
    INPUT_MODE=file \
    INPUT_WAV=../samples/user_utterance.wav \
    OUTPUT_WAV=../samples/bot_response_gpu.wav \
    FILE_MODE_DRAIN_S=15 \
    FILE_MODE_IDLE_TIMEOUT_S=45 \
    python3 pipecat_demo.py'
  scp e3-vast:/workspace/samples/bot_response_gpu.wav ./samples/
  ```
  **Note**: `SSL_CERT_FILE` is needed because Mac Python 3.13 ships without root CAs. On the remote box it may not be needed but it doesn't hurt.
  **bot_response_gpu.wav must contain bot-only audio now** (the AudioBufferProcessor merge bug is fixed in this commit). If `BotAudioRecorder captured 0 bytes` shows in the log, the TTS pipeline is silent — investigate before declaring success.

- [ ] **4. End-to-end latency** — ALREADY CAPTURED tonight. `metrics_gpu.json` has the canonical Pipecat `UserBotLatencyObserver` reading: 916 ms total (Groq 652 ms + megakernel 36 ms + 228 ms pipeline). No further work needed unless you want a higher-n version.

- [ ] **4b. Fix audio intelligibility (NEW — tomorrow's biggest item)** — see Agent C's diagnosis in tonight's chat history. Concrete next steps:
   1. Inspect `/workspace/qwen3-tts-1.7b/generation_config.json` for the actual `bos_token_id` and `eos_token_id` (don't trust the hardcoded 2150). Likely there's a separate audio-side BOS like `<|audio_bos|>` distinct from text BOS.
   2. In `megakernel_tts.py:generate()` around line 400, replace `prev_tok = 0` with the audio-BOS token id from the generation_config.
   3. Read the upstream `Qwen3TTSForConditionalGeneration.generate()` from `transformers==4.57.3` (likely in `transformers.models.qwen3_tts.modeling_qwen3_tts`) and copy its prompt-building logic: chat template with `<|im_start|>system\n...<voice spec for "ryan">...<|im_end|><|im_start|>user\n{text}<|im_end|><|im_start|>assistant\n<|audio_bos|>`.
   4. In `_talker.prefill_text(text, ...)` replace the bare-text path with the chat-templated path. Pass `add_special_tokens=True`.
   5. Re-bench TTFC + RTF — should not regress since the model is the same, just conditioning is different. Bot output should now emit EOS naturally + the WAV should be ~3-5 sec of intelligible speech, not 168 sec of babble.
   Budget estimate: ~2 h on GPU at $1.336/hr = ~$2.70. **Note this is over our $2.30 remaining headroom** — if it goes long, fall back to demo option (b) using mac_say substitute.

- [ ] **5. Record the demo** — script at `~/brain/build/side-projects/e3-video-recording-script.md` has the narration + test sentences. ~5 min final video. Tools: QuickTime screen recording with audio.

- [ ] **6. Upload video** (Loom unlisted, YouTube unlisted, or wherever) — paste URL into `~/brain/build/side-projects/e3-submission-email-draft.md` `<VIDEO_LINK>` placeholder.

- [ ] **7. Final commit** — `git add -A && git commit -m "Final: end-to-end Pipecat bench + demo video" && git push`.

- [ ] **8. Stop GPU** — `bash scripts/deploy.sh --stop` (preserves disk). Or `--destroy` if you're sure nothing else needs the box.

- [ ] **9. Send the email** — content in `~/brain/build/side-projects/e3-submission-email-draft.md`. Recipient: the Contrario contact thread.

- [ ] **10. Reimbursement** — receipts in `~/brain/build/side-projects/e3-contrario-reimbursement-receipts.md`. Submit per Contrario's "save your receipt and send it to us" line in the original email.

---

## Path B — actionable checklist (if shipping as-is)

- [ ] **1.** Edit `~/brain/build/side-projects/e3-submission-email-draft.md`, replace `<VIDEO_LINK>` with: *"Demo video not yet recorded — happy to do a live walkthrough on the Caleb chat if useful."*
- [ ] **2.** Send.
- [ ] **3.** `bash scripts/deploy.sh --destroy` to tear down the GPU instance and stop the $0.02/hr disk charge.
- [ ] **4.** Reimbursement request per receipts file.

---

## Findings — what's true at this checkpoint

### Performance (n=5 with 3 warmup, single RTX 5090 sm_120)

**Config A** — sine-stub codec, isolates the megakernel hot path:

| Metric | Value | vs brief tightest | vs brief loosest |
|---|---|---|---|
| TTFC | **17.2 ± 0.02 ms** | <50 ms ✅ | <90 ms ✅ |
| RTF | **0.209 ± 0.0001** | <0.1 ❌ | <0.3 ✅ |
| Talker decode-only | **503.1 tok/s** | — | — |

**Config B** — REAL Qwen3-TTS codec (271 weights, clean-room reimpl), first run:

| Metric | Value | Verdict |
|---|---|---|
| TTFC | **694 ms** | misses all tiers — JIT-compile dominated |
| RTF | **0.347** | misses all tiers |
| Audio | broadband, voiced/unvoiced spectrum (centroid 976 Hz, dynamic range 23.9 dB) | real synthesis, not beep |

**Baseline reproduction**: stock AlpinDale qwen_megakernel on Qwen3-0.6B = **1034.6 tok/s** (matches their published 1036.3 within noise).

**KV cache correctness**: 5/5 PASS — determinism, monotonic positions, no OOR tokens, prompt conditioning, reset semantics.

**API health**: Deepgram ✅, Groq llama-3.1-8b-instant ✅, HF token (user `pratham7711`, fine-grained) ✅.

### Bugs found + fixed this session
| # | Severity | Where | What was wrong | Fix |
|---|---|---|---|---|
| 1 | **CRITICAL** | `qwen3_tts_components.py:99, 781-810` | Semantic codebook hardcoded 2048 — upstream config has 4096. The `embedding_sum` weight silently landed in `unexpected_keys` and the codebook stayed at random init. **This is the real reason audio was garbled (independent of text prefill).** | Plumb `semantic_codebook_size=4096` through `_SplitResidualVectorQuantizer` |
| 2 | HIGH | `megakernel_tts.py:222` | `load_components()` returns 3-tuple after the real-codec rewrite; was unpacking 2 → silently flipped to STUB | Unpack 3-tuple |
| 3 | HIGH | `bench_megakernel.py:169-170` | Decoder was called with `model_name=` (wrong kwarg); also called `.tokenizer.encode(...)` which doesn't exist | Use `model_path=`, seed with token id 0 |
| 4 | HIGH | `ui_v2.py:303` | TTFC timer started AFTER prefill → undersold the metric vs brief definition | Start timer BEFORE prefill, sync once |
| 5 | HIGH | `qwen3_tts_components.py:671` | `_sliding_mask()` allocated dense (T,T) float -inf tensor every forward call → 80 s hangs at T≥80 | Return bool mask (~1 KB), SDPA short-circuits |
| 6 | MED | `pipecat_demo.py:WavFileInputProcessor` | Used `.start()` hook removed in Pipecat 1.3.0 → pump never fired → e2e test timed out idle | Kick pump from `process_frame` on first `StartFrame` |
| 7 | MED | Mac Python 3.13 stdlib | No root CAs by default → Deepgram/Groq HTTPS handshake fails with `SSL: CERTIFICATE_VERIFY_FAILED` | `export SSL_CERT_FILE="$(python3 -c 'import certifi; print(certifi.where())')"` before running |
| 8 | MED | `ui_v2.py:prepend_history` | Truthy check on a pandas DataFrame raises `ValueError` | Route via `.empty` attribute |
| 9 | **CRITICAL** | `pipecat_demo.py:AudioBufferProcessor` | Stock processor merges user + bot frames into ONE buffer. With stub TTS = silence, the saved `bot_response.wav` was the user's input echoed back — masquerading as bot output. The "smoke test validates TTS plumbing" claim was hollow. | New `BotAudioRecorder` FrameProcessor captures `TTSAudioRawFrame` only. Saved WAV is now genuinely bot-only. |
| 10 | HIGH | `pipecat_demo.py:run` | `load_dotenv(override=True)` blocked shell env from overriding `.env` — couldn't switch to stub from CLI. | `override=False` |
| 11 | HIGH | `megakernel_tts.py:__init__` | Silent fallback to stub on Decoder / load_components failure. Hid real init errors. | Removed try/except — failures RAISE. Opt-in stub via `stub=True` only. |
| 12 | MED | `pipecat_demo.py:PipelineTask` | Default 5-min idle timeout made file-mode smoke tests painfully slow. | `idle_timeout_secs=30` + configurable drain via `FILE_MODE_DRAIN_S`. |
| 13 | MED | `megakernel_tts_service.py` | `TTSSettings: NOT_GIVEN model/voice/language` validator error every run. | Pass `model/voice/language` to `super().__init__()`. |
| 14 | NEW FEATURE | `megakernel_tts.py:_mac_say_generate` | No way to demo real bot audio on Mac (only silence or GPU). | Added `MEGAKERNEL_STUB=mac_say` mode: macOS `say` → 24 kHz int16 PCM → 80 ms frame chunks matching the real codec shape. End-to-end Mac demo confirmed. |
| 15 | HIGH (UI) | `ui_loopback.py` | Gradio 6.15.2 vs 4.x target: `editable=True` default put mic widget into WaveSurfer trim mode (no Play button); `gr.Dataframe` silently dropped `list[list[str]]` returns. | `editable=False, interactive=True` on mic; dict return for Dataframe; `autoplay=True` on every output; Clear buttons; `show_progress="full"`. Verified PASS via Playwright. |

### Architecture decisions that paid off
- **Scope megakernel to the talker only** (per brief) — keeping code_predictor + codec in PyTorch was the right call; let us iterate without rebuilding the kernel
- **Two-config benchmark reporting (A: stub, B: real)** — isolates kernel performance from codec overhead; reviewer sees both honestly
- **`vastai` CLI over Playwright** — once the API key was in `~/.vast_api_key`, restart/stop/show became ~2 sec instead of ~60 sec navigation
- **Clean-room codec reimpl (avoiding the `qwen-tts` pip package)** — sidestepped the torchaudio ABI conflict with PyTorch 2.10.0a NGC. All 271 weights load with 0 unexpected.

### What's still rough / honest gaps
- **MRoPE in the kernel is single-axis collapse** — math-equivalent to vanilla 1D RoPE @ θ=1M for our autoregressive-only path. Full multi-axis would matter if we wired multi-modal prompts.
- **Codec Config B cold-compile** — first call is ~694 ms TTFC because the codec's 8-layer transformer + ConvNet decoder JIT-compile each kernel separately. With a `torch.compile` pass over the codec or proper warmup, should drop substantially.
- **Demo video not recorded** — last open deliverable.
- **End-to-end mic→audio latency not measured** — last open metric. Path A item 4 covers this.

---

## State of the world

### What's where on disk
| Asset | Location |
|---|---|
| Modified megakernel | `qwen_megakernel_modified/{csrc/kernel.cu, qwen_megakernel/{build,model,bench}.py}` |
| Inference server | `inference-server/{megakernel_tts.py, megakernel_tts_service.py, qwen3_tts_components.py, ui_v2.py, ui_loopback.py, pipecat_demo.py, bench_megakernel.py, validate_keys.py}` |
| Real-codec wrapper | `inference-server/qwen3_tts_components.py` (271 weights, clean-room) |
| Pipecat demo (Groq + Deepgram + INPUT_MODE=mic\|file) | `inference-server/pipecat_demo.py` |
| Gradio UI v2 | `inference-server/ui_v2.py` |
| Mac-local loopback UI (mic + API tests, no GPU) | `inference-server/ui_loopback.py` |
| Bench results JSON | `bench_results.json`, `bench_megakernel_talker.json` |
| Sample WAVs | `samples/{user_utterance, test_01_short, test_02_numerics, test_03_freight, demo_intro, bot_response}.wav` |
| Visuals | `docs/img/{spectrum_real_codec, perf_vs_brief, ui_screenshot}.png` |
| One-shot deploy script | `scripts/deploy.sh` |
| Repo docs | `README.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `NOTICES.md`, `Makefile`, `docs/spectrum_stats.md` |

### What's in brain (`~/brain/build/side-projects/`)
| File | Purpose |
|---|---|
| `project-e3-megakernel-tts.md` | **Main hub** — FINAL STATE block, sprint history, decisions log, deliverables, risk register |
| `e3-kernel-mod-plan.md` | Line-by-line kernel.cu / model.py / build.py / bench.py edits |
| `e3-mrope-math.md` | MRoPE math + CUDA pseudocode + single-axis collapse argument |
| `e3-pipecat-integration-notes.md` | Pipecat design notes — TTSService shape, frame conventions |
| `e3-manager-audit.md` | First-pass audit (manager-style review with action plan) |
| `e3-final-code-review.md` | Code-reviewer agent pass #1 (HIGH/MED/LOW findings) |
| `e3-final-code-review-2.md` | Code-reviewer agent pass #2 (the codebook 4096 finding) |
| `e3-video-recording-script.md` | Demo video script — narration, test sentences, fail-safes |
| `e3-submission-email-draft.md` | Send-ready email (only `<VIDEO_LINK>` placeholder left) |
| `e3-contrario-reimbursement-receipts.md` | $7.70 spend breakdown for reimbursement request |

### Live state at checkpoint
- **No claude-spawned processes alive** — all UIs/agents/monitors stopped or expired
- **Vast GPU**: stopped (`actual=exited`, `intended=stopped`)
- **Mac SSH tunnel** to port 8080: dead (intentionally, GPU is down)
- **Mac caffeinate** daemons (4 of them, pre-existing + one I started): harmless, will exit on their own timers
- **Brain peer sync** (pratham7711/pratham-brain): up to date

---

## Environment cheat-sheet

### Python on Mac
- System `python3` is 3.9 — most deps don't install there
- Use `/tmp/vastvenv/bin/python3` (Python 3.13 venv) for anything touching `groq`, `gradio`, `soundfile`, `vastai`, `huggingface-hub`
- For HTTPS calls on Mac Python 3.13: `export SSL_CERT_FILE="$(python3 -c 'import certifi; print(certifi.where())')"`

### Vast CLI
- API key at `~/.vast_api_key`
- Binary: `/tmp/vastvenv/bin/vastai`
- Common: `vastai show instances`, `vastai start/stop instance <id>`, `vastai destroy instance <id>`, `vastai search offers --raw ...`

### SSH
- Host alias `e3-vast` in `~/.ssh/config` (local only, not in any repo)
- Key: `~/.ssh/id_ed25519_e3vast` (separate from your GitHub key)
- When GPU restarts, `scripts/deploy.sh` rewrites HostName + Port in the config block automatically

### Brain
- `~/brain/.scripts/sync.sh pull` before reading, `push` after writing
- Repo: `pratham7711/pratham-brain` (private GitHub)

### .env (gitignored)
- At `inference-server/.env` — contains DEEPGRAM_API_KEY, LLM_API_KEY (Groq), HF_TOKEN
- Template: `inference-server/.env.example`

---

## Companion notes — read these for deeper context
- `~/brain/build/side-projects/project-e3-megakernel-tts.md` — main hub
- `~/brain/build/side-projects/e3-final-code-review-2.md` — last code review (consult if anything looks suspicious in code)
- `~/brain/build/side-projects/e3-video-recording-script.md` — for the demo recording
- `~/brain/build/side-projects/e3-submission-email-draft.md` — the email
- `ARCHITECTURE.md` (in this repo) — deeper companion to README's mermaid diagram

---

## Resume marker location
`/Users/pratham/Documents/Repositories/e3-megakernel-tts/RESUME_HERE.md` — this file. Always overwrite this in place; don't fork copies.
