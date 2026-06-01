# Demo Script — e3 take-home

~2 minutes spoken. Video shows the Gradio UI on the left, GPU-box terminal log on the right.

---

## Opening (20s)

> "This is my submission for the e3 take-home — wiring AlpinDale's `qwen_megakernel` as the audio decode backend for Qwen3-TTS-1.7B-CustomVoice, streaming through Pipecat. The headline target the brief weights highest is intelligible English audio coming out the speaker; the perf benchmarks are graded in three tiers around that.
>
> I'll show you the UI, take you through a mic-in / speaker-out turn, then walk through the engineering. The most useful slide is what I learned about the megakernel↔Qwen3-TTS wiring — that's where the real work was."

## Live demo (40s)

*[Browser opens to localhost:8080. Status pill at top: "🎙️ Mic ready · 🔊 Speaker armed" — page-load `getUserMedia` already prompted and was granted before recording started.]*

*[record into mic: "What's the weather like tomorrow?" — click Send]*

*[bot WAV streams in browser audio element, autoplays, voice says back a Groq-llama answer through Qwen3-TTS]*

> "Notice three things happening at once: STT transcript appears mid-stream, the LLM tokens stream in, and the TTS audio starts playing while the talker is still decoding the rest of the utterance. That's the streaming property the brief flags as make-or-break — no end-of-utterance buffering.
>
> Also — the audio that came out is intelligible English. Earlier in the day, the same pipeline produced 5 seconds of speech-shaped broadband sound with no phonetic content. I'll show you why."

## What I actually did (40s) — the engineering arc

> "The first pass passed the brief's perf benchmarks: TTFC 18 milliseconds, RTF 0.12. I almost shipped it. But before recording the demo I ran the generated audio back through Deepgram nova-2 — same STT model as our production input path, used here as a quality gate — and Deepgram returned blank transcripts at 0.0 confidence on every test.
>
> That surfaced the real bug. I spawned a dedicated wiring-audit agent that diffed our implementation against vanilla upstream `Qwen3TTSForConditionalGeneration` and found four stacked bugs:
>
> One: the Code Predictor was being driven by a heuristic int-token lookup. Upstream feeds the talker's last-layer hidden state into a 5-layer transformer that AR-decodes 15 additional codebooks per frame.
>
> Two: the talker's next AR step was receiving only codebook-0's embedding. Upstream feeds the sum of all 16 codebook embeddings from the previous step.
>
> Three: we were greedy-argmaxing everywhere; the model is trained with sampling — `do_sample=True, top_k=50, temp=0.9, rep_penalty=1.05`. Greedy collapses to a babble attractor.
>
> Four: the audio prefix was sequence-concatenated. Upstream does element-wise sum of text projection and codec embeddings at matching positions — 10 prefix positions, not 22.
>
> I implemented all four. To confirm wiring vs codec, I also ran vanilla upstream Qwen3-TTS on the same GPU as a baseline — Deepgram transcribes that at 0.9995. Our megakernel-wired path now matches that baseline."

## The numbers — measured (25s)

> "Audio QA — Deepgram nova-2 round-trip on the megakernel output:
>
> - 'Hello. How are you doing today?' → 'Hello. How are you doing today?' at **1.000 confidence**.
> - Matches vanilla upstream Qwen3-TTS (the control) at 0.9995.
>
> Perf — n=5, 3 warmup, RTX 5090 NGC PyTorch 2.10.0a, explicit `cuda.synchronize` at every timer boundary:
>
> - **TTFC: 25.32 ± 0.03 ms** — passes ALL three brief tiers (Tightest <50 ✅, Perf <60 ✅, Deliverables <90 ✅).
> - **RTF: 0.1452 ± 1.7e-4** — passes the **Performance tier** (<0.15 ✅) and Deliverables (<0.30 ✅). Misses Tightest (<0.10) by 0.045.
> - Per-utterance decode wall: 743 ms for 5.12 s of audio.
>
> The brief's headline targets — TTFC <60 ms and RTF <0.15 — are both green. The pre-megakernel-AR baseline was RTF 0.181, missing Perf tier by 0.03; the kernel surgery I'll talk about next closed that gap."

## The actual megakernel finally running (15s)

> "The most load-bearing thing I want to flag — and the reason RTF dropped from 0.181 to 0.145 today — is that BEFORE this morning's session, the persistent megakernel wasn't actually in the production AR hot path. `Decoder.step(token_id)` (the original AlpinDale entry point) was only being called by bench harnesses. The actual TTS path was going through a CUDA-graph-captured PyTorch 28-layer forward.
>
> The reason for the bypass was architectural: the megakernel's `_decode` takes an int token id and does the embedding lookup internally. But Qwen3-TTS's AR step needs a precomputed input embedding — `last_id_hidden + sum(16 codebook embeddings) + trailing_text` — there's no token-id surface for that.
>
> The fix was ~80 lines across `csrc/kernel.cu` + `csrc/torch_bindings.cpp` + `qwen_megakernel/model.py`: added a nullable `input_embed: const __nv_bfloat16*` parameter to `ldg_decode_kernel_direct`, a new `launch_ldg_decode_direct_embed` launcher that skips the fused lm_head (because our sampling tail can't live in the kernel), a `decode_embed` torch op registration, and a `Decoder.step_embed_megakernel()` method that's now the default path in `step_embed`. JIT-recompile via `torch.utils.cpp_extension.load` is incremental — took 8.6 seconds.
>
> Result: ~280 graph-replay ops collapse to ONE persistent megakernel launch per AR step. Same math (bf16 RMSNorm, same MRoPE table, same SDPA, same final norm), just one launch instead of hundreds. Deepgram still 1.000."

## Honest gaps + path forward (15s)

> "Remaining gap to the brief's Tightest tier (<0.10 RTF) is 0.045. Next lever: collapse CP's 14 per-step CUDA-graph replays + Gumbel sampling into one megagraph. Estimated 3-5 ms per AR step, projects to RTF ~0.13. Scoped in the CHANGELOG, not built this session — would be another 60-90 minutes of focused work.
>
> The other thing worth naming honestly: the megakernel's MRoPE table-build code is in place but collapses to vanilla 1D RoPE @ θ=1e6 in the autoregressive-only path, because all three axes share a single position counter post-prefill. That's mathematically equivalent for what we do (audio-only AR after text prefill), so it's not a correctness gap — but a multi-modal extension would need to wire distinct per-axis positions."

## Closing (10s)

> "Repo is public, README has the full performance + audio-QA tables with methodology, bench is reproducible with `make bench`, `scripts/deepgram_stt_check.sh` is the QA gate that gates every audio change. Email and reimbursement receipts to follow."

---

## Production notes (for recording)

- **Browser**: Chrome at `http://localhost:8080` (Gradio 6, custom UI v2). Status pill should already show "🎙️ Mic ready · 🔊 Speaker armed" green — page-load JS triggered `getUserMedia`. If Safari, do one click anywhere first to unlock autoplay.
- **Tunneling**: `ssh -f -N -L 8080:localhost:8080 e3-vast` running before opening browser.
- **Recording tool**: QuickTime screen recording with audio. Mac internal mic + system audio capture (Soundflower / built-in `Loopback` if installed).
- **Mic distance + room**: quiet room, 4-6 inches from mic, normal speaking voice.
- **Failsafe**: if live mic loop has any hiccup (Deepgram WS error, etc.), pre-generated `samples/bot_test_polish_4_aiden_med.wav` is the same megakernel-wired output produced from a file input — play it through Mac speakers as a fallback shot.
- **Length cap**: 2-3 minutes total. The "what I did" section is the highest-leverage 40s — don't compress it.
- **Alternative path** (if browser UI hits any issue): `scripts/run_voice_turn.sh <user-wav> <bot-wav>` does the full mic→GPU→speaker round-trip via Pipecat file-mode in one command, plays the bot WAV through `afplay`. Same backend, different transport. Still shows everything the brief asks for.
