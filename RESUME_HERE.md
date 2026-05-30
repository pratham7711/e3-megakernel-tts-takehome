# Resume marker — e3 take-home (2026-05-30 19:55 IST)

> **Start here every new session.** Single source of truth for what's done, what's left, and how to continue.

---

## TL;DR — where we are

- **Repo**: https://github.com/pratham7711/e3-megakernel-tts-takehome (public, HEAD `334349c`)
- **Status**: ~95% done. **Last open item**: live GPU run + demo video.
- **Deadline**: 2026-06-01 EOD (≈ 24 h from this marker)
- **Spend so far**: ~$7.70 of $10 reimbursement budget. Budget headroom: ~$2.30 (≈ 75 min GPU at $1.755/hr).
- **GPU**: instance `#38548758` is STOPPED. Disk preserved; one `vastai start instance 38548758` brings it back.
- **Email**: drafted, only the `<VIDEO_LINK>` placeholder remains. Send-ready otherwise.

---

## Decision the human needs to make first

**Path A (recommended)** — Finish the brief 100%: restart GPU once (~45 min), run live demo, record video, send email. ETA ~1 h wall, ~$1.30 GPU.

**Path B** — Ship as-is. Open `~/brain/build/side-projects/e3-submission-email-draft.md`, swap `<VIDEO_LINK>` for "happy to record live on a call", send. ETA ~5 min.

Default to **Path A** unless the user explicitly says otherwise. Reasoning: video is an explicit brief deliverable; budget is comfortable; the only real cost is ~45 min of focused work.

---

## Path A — actionable checklist

Run these from `~/Documents/Repositories/e3-megakernel-tts/`.

- [ ] **1. Restart GPU + redeploy** — single command does everything (start instance, wait for SSH, update `~/.ssh/config`, rsync code, rebuild kernel JIT, launch UI on remote, set up local Mac tunnel):
  ```bash
  bash scripts/deploy.sh
  ```
  If the host has no free slot (`Required resources are currently unavailable`), retry every minute for ~5 min. If still stuck, the script's `--destroy` flag tears the old instance + you rent fresh: `/tmp/vastvenv/bin/vastai search offers --raw 'gpu_name=RTX_5090 num_gpus=1 verified=True rentable=True reliability>0.99 inet_down>1000 disk_space>40' -o 'dph_total asc' | head` then `vastai create instance <id> --image nvcr.io/nvidia/pytorch:26.01-py3 --disk 80 --ssh`.

- [ ] **2. Verify in browser** — open http://localhost:8080. Click Generate with the default text. **Expect**: audio should sound DIFFERENT from the previous run (this session committed the **semantic codebook 4096 fix** — the prior runs produced garbled audio because the semantic codebook was at random init). If the new audio still sounds broken, see `~/brain/build/side-projects/e3-final-code-review-2.md` for the next 8 LOW-severity findings.

- [ ] **3. Run the live Pipecat e2e** — proves the brief's Step 3 and gives us the missing 4th metric (end-to-end latency):
  ```bash
  ssh e3-vast 'cd /workspace/inference-server && \
    SSL_CERT_FILE="$(python3 -c "import certifi; print(certifi.where())")" \
    INPUT_MODE=file \
    INPUT_WAV=../samples/user_utterance.wav \
    OUTPUT_WAV=../samples/bot_response_gpu.wav \
    python3 pipecat_demo.py'
  scp e3-vast:/workspace/samples/bot_response_gpu.wav ./samples/
  ```
  **Note**: `SSL_CERT_FILE` is needed because Mac Python 3.13 ships without root CAs. On the remote box it may not be needed but it doesn't hurt.

- [ ] **4. Capture end-to-end latency** — measure (mic-WAV-input timestamp) → (first PCM byte timestamp on output WAV). Add it to `bench_results.json`. Update README's Performance table to fill the "end-to-end latency" cell that's currently blank.

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
