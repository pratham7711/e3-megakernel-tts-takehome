# Resume marker — 2026-05-30 19:30 IST

## Last directive from user
> "record what is done in brain in detail this session is getting a lot slow we'll start new session"

Prior in-session asks (still relevant for next session):
1. Build a browser UI that takes mic → STT → LLM → TTS → speaker round-trip.
2. Test without GPU first (loopback + API health).
3. Verify Deepgram / Groq / HF tokens all work.
4. Finish the e3 take-home: live Pipecat run + demo video + final commit + email submission.

## State when we stopped
- **Domain**: build / career — e3 Group take-home (via Contrario)
- **Repo HEAD**: `3adb7c9` — "Code review #2 fixes + repo polish + visuals + sample WAVs"
- **Brain HEAD** (last sync): up to date with origin
- **Public repo**: https://github.com/pratham7711/e3-megakernel-tts-takehome (master)
- **GPU instance #38548758**: **STOPPED** ($1.7556/hr paused; disk-only ~$0.02/hr)
- **Spend so far**: ~$7.70 of $10 budget
- **Submission deadline**: 2026-06-01 EOD (~24 hours from resume time)

## Live work crossing the /clear boundary

### Background processes still running on Mac
- **caffeinate PIDs**: 50126, 68454, 88073, 92226 — leave them. (88073 = my 8-hour `-di` started this session; the others predate this session.) Will all exit on their own timers.
- **Gradio loopback UI** (`ui_loopback.py` PID 84739) — **killed at preclear** (was at port 7861, user reported "no mic found" — likely macOS Privacy → Microphone gate). Re-launch with `/tmp/vastvenv/bin/python3 inference-server/ui_loopback.py` next session.
- **SSH tunnel Mac:8080 → remote:8080** — port 8080 was free at preclear; tunnel had already died when GPU went down.

### Sub-agents
- All 11 sub-agents dispatched this session have **completed**. Outputs visible on disk + in brain notes.

### Monitors
- All Monitor tool watchers expired with the session — none armed at /clear.

### Scheduled wakeups / Cron / RemoteTrigger
- **None** created this session.

### How to verify post-/clear
```bash
ps -A -o pid,etime,command | grep -E "ui_loopback|caffeinate|ssh.*8080" | grep -v grep
ls -lh ~/Documents/Repositories/e3-megakernel-tts/samples/bot_response.wav  # smoke test output (~237 KB)
/tmp/vastvenv/bin/vastai show instances --raw | python3 -c "import json,sys; t=sys.stdin.read(); i=t.find('['); d=json.loads(t[i:]); print(d[0].get('actual_status'))"
```

## Files written / changed this session (recap)

### Modified (committed)
- `qwen_megakernel_modified/csrc/kernel.cu` — HIDDEN=2048, INTERMEDIATE=6144, VOCAB=3072, LM head retuned
- `qwen_megakernel_modified/qwen_megakernel/model.py` — talker.model.* loader, MRoPE table, `prefill_text()`
- `qwen_megakernel_modified/qwen_megakernel/build.py` — flag defaults reverted to high-block-count
- `inference-server/megakernel_tts.py` — streaming yield, text_prefill, torch.compile, 3-tuple unpack fix
- `inference-server/megakernel_tts_service.py` — Pipecat TTSService subclass
- `inference-server/qwen3_tts_components.py` — REAL Qwen3-TTS codec (271 weights, clean-room) + semantic codebook 4096 fix + bool sliding mask
- `inference-server/ui_v2.py` — polished dashboard, TTFC includes prefill, DataFrame defensive
- `inference-server/bench_megakernel.py` — Decoder kwargs + seed token fix
- `inference-server/pipecat_demo.py` — Groq, INPUT_MODE=mic|file, VAD, WavFileInputProcessor
- `inference-server/validate_keys.py` — Deepgram/Groq/HF SDK probes (all PASS)
- `inference-server/requirements.txt` — pinned
- `README.md` — mermaid diagram, decisions log, dry-run-verified how-to-run
- `inference-server/.env.example` — Groq + HF_TOKEN + INPUT_MODE schema

### Added (committed)
- `ARCHITECTURE.md`, `CHANGELOG.md`, `NOTICES.md`, `Makefile`
- `docs/img/{spectrum_real_codec,perf_vs_brief,ui_screenshot}.png`
- `docs/spectrum_stats.md`
- `scripts/{deploy.sh,make_spectrum_chart.py,make_perf_chart.py}`
- `samples/{user_utterance,test_01_short,test_02_numerics,test_03_freight,demo_intro}.wav`
- `bench_megakernel_talker.json`, `bench_results.json`, `bench_audio.wav`, `demo_audio_sample.wav`

### Pending uncommitted
- `inference-server/ui_loopback.py` — new (Gradio loopback UI for mic+API testing)
- `samples/bot_response.wav` — output from the Pipecat smoke test that succeeded

## Numerical findings (honest, n=5 with 3 warmup)

### Config A — sine-stub codec (megakernel isolation)
- **TTFC = 17.2 ± 0.02 ms** — PASS all 3 brief target tiers (<50 / <60 / <90 ms)
- **RTF = 0.209 ± 0.0001** — PASS deliverables tier (<0.3); misses tighter
- Talker-only decode: **503.1 tok/s** (vs AlpinDale baseline 1034.6 tok/s on 0.6B)
- Wall / 2 s audio: 418.4 ms

### Config B — REAL Qwen3-TTS codec (cold start, first run)
- **TTFC = 694 ms** — misses all 3 target tiers (JIT-compile dominated; under investigation post-codec-fix)
- **RTF = 0.347** — misses all 3 target tiers
- Audio: real broadband output (spectral centroid 976 Hz, 23.9 dB dynamic range, formant-like)

### KV cache correctness — 5/5 PASS
determinism, monotonic position, no OOR tokens, prompt-conditioned outputs, reset() clears

### API health
All 3 PASS (Deepgram, Groq llama-3.1-8b-instant, HF whoami pratham7711)

### Pipecat smoke test result
`samples/bot_response.wav` = 237 KB, **4.95 s, 24 kHz mono PCM_16, peak amp 0.48** — written by `pipecat_demo.py INPUT_MODE=file` agent run. **End-to-end pipeline executed and produced audible WAV output. Validates Deepgram + Groq + TTS through the Pipecat plumbing.**

#### Two gotchas surfaced — both fixed and committed
1. **SSL cert verify on Python 3.13** (Mac homebrew Python ships without root CAs). Mac-only run-command preamble: `SSL_CERT_FILE="$(python3 -c 'import certifi; print(certifi.where())')"`.
2. **`Pipecat 1.3.0` removed `FrameProcessor.start()`**. Our `WavFileInputProcessor` now kicks off the pump inside `process_frame` when the first `StartFrame` arrives. Future-self: do NOT add a `.start()` override to a Pipecat 1.3+ FrameProcessor.

Full repro command:
```bash
cd inference-server
SSL_CERT_FILE="$(python3 -c 'import certifi; print(certifi.where())')" \
  MEGAKERNEL_STUB=1 INPUT_MODE=file \
  INPUT_WAV=../samples/user_utterance.wav OUTPUT_WAV=../samples/bot_response.wav \
  python3 pipecat_demo.py
```

## Bugs found + fixed this session
- **H1 (CRITICAL)**: `_SplitResidualVectorQuantizer` hardcoded codebook_size=2048 for BOTH semantic and acoustic. Upstream config: semantic_codebook_size=4096. The semantic codebook stayed at random init → garbled audio. Plumbed `semantic_codebook_size=4096` through.
- **H1 (prior pass)**: `megakernel_tts.py` `load_components()` was unpacking 2-tuple from 3-tuple result → silently flipped to STUB.
- **H2**: TTFC timer was started AFTER prefill → undersold. Now starts before.
- **H2 (prior)**: `bench_megakernel.py` passed `model_name=` to Decoder (wrong kwarg) + called `.tokenizer.encode()` which doesn't exist.
- **H3**: TTFC measured at batched-flush time not first frame. Fixed.
- **Codec O(T²)**: `_sliding_mask()` allocated dense (T, T) float -inf tensor each forward pass → 80s hangs at T>=80. Now returns bool mask (~1 KB).
- **DataFrame truthiness**: `prepend_history()` crashed on pandas DataFrame `or` semantics. Fixed defensively.
- **`<this-repo>` placeholder + 3 other README dry-run failures** — all fixed.

## Pending decisions
**None blocking** — submission email draft is ready, only `<VIDEO_LINK>` placeholder remains.

The only judgment call left for the user: **do you want to finish the live demo run (~45 min + ~$1.30 GPU) before emailing, or ship as-is with honest "not yet recorded" disclosures in the email?**

## Next action — when ready to resume

### Path A — Finish the brief 100% (recommended)
```bash
cd ~/Documents/Repositories/e3-megakernel-tts
bash scripts/deploy.sh                       # restart GPU, rsync, rebuild kernel, launch ui_v2
# wait for "open http://localhost:8080" message, then:
# 1. open UI in browser, click Generate to verify codec-4096 fix produces better audio
# 2. run: make demo-stub                     # validates pipecat_demo INPUT_MODE=file e2e
# 3. screen-record QuickTime: 4-5 min walkthrough per `~/brain/build/side-projects/e3-video-recording-script.md`
# 4. upload video, paste URL into `~/brain/build/side-projects/e3-submission-email-draft.md`
# 5. git add -A && git commit -m "Final: demo video + e2e bench" && git push
# 6. bash scripts/deploy.sh --stop           # stop GPU
# 7. send the email
```

### Path B — Ship as-is
Open `~/brain/build/side-projects/e3-submission-email-draft.md`, swap `<VIDEO_LINK>` for "not yet recorded due to budget/time constraints — happy to record live on a call", and send.

## Resume marker location
`/Users/pratham/Documents/Repositories/e3-megakernel-tts/RESUME_HERE.md` (this file)

## Companion brain notes
- `~/brain/build/side-projects/project-e3-megakernel-tts.md` — main hub
- `~/brain/build/side-projects/e3-final-code-review-2.md` — last code review findings (use as todo list if anything looks stale)
- `~/brain/build/side-projects/e3-submission-email-draft.md` — email draft (just needs video URL)
- `~/brain/build/side-projects/e3-contrario-reimbursement-receipts.md` — $7.70 spend breakdown for the reimbursement request
- `~/brain/build/side-projects/e3-video-recording-script.md` — demo script with timing, test sentences, fail-safes
