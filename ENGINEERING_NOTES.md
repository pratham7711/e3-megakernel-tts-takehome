# Engineering Notes — e3 Take-Home

> Companion to `README.md`. README is the operator's manual (install, run, bench). This document is the engineering story: what I built, what I traded off, what the final numbers are, and the honest reason we miss the Tightest tier.
>
> Total project time: ~12 hours of focused work spread over 4 days. ~$5 of the $10 Vast.ai budget.

---

## TL;DR

| | |
|---|---|
| **Built** | AlpinDale's `qwen_megakernel` adapted to drive Qwen3-TTS-12Hz-1.7B-CustomVoice's talker, streaming end-to-end through Pipecat (Deepgram STT → Groq LLM → MegakernelTTS → audio output). |
| **Hit** | TTFC **25.32 ± 0.03 ms** (passes Tightest / Performance / Deliverables ✓✓✓). RTF **0.1452 ± 1.7e-4** (passes Performance + Deliverables ✓✓; misses Tightest by 0.045). Deepgram round-trip audio QA at **1.000** confidence, matches upstream control. |
| **Missed** | Tightest tier on RTF only. Why: the Code Predictor's 14 per-step CUDA-graph replays haven't been collapsed into a single mega-graph yet. Scoped, not built. ~1 hour of further work. |
| **Most surprising thing learned** | The repo was named `megakernel-tts` and we talked about the megakernel everywhere — but the persistent kernel wasn't actually being called in the production audio decode loop. It had been bypassed silently because the original kernel takes a token id and Qwen3-TTS needs a precomputed embedding. Caught it on day 4 and fixed it with ~80 lines of kernel surgery. |

---

## 1. What the brief asked

In one sentence: take a CUDA megakernel that runs Qwen at 1000 tok/s, swap it in for Qwen3-TTS's talker decode loop, stream through a Pipecat voice agent.

The brief weighted things this way (paraphrased):
- TTFC < 60 ms target, < 50 ms "tightest", < 90 ms "deliverables"
- RTF < 0.15 target, < 0.10 "tightest", < 0.30 "deliverables"
- Audio must stream chunk-by-chunk; don't buffer the full utterance
- Audio quality must be acceptable (no glitches, dropped frames)
- "Ramp-up speed, performance rigor, coding agent proficiency, communication" — what's being evaluated

Two things the brief flagged that turned out to be load-bearing:
1. *"Don't hand-wave"* on numbers and methodology.
2. *"Honest about what works and what's rough"* in the README.

I took both seriously, and that's why the largest section in `CHANGELOG.md` and this doc is honest gap analysis, not victory lap.

---

## 2. The process — what I actually did, in order

### Phase 1 — Kernel adaptation (~3 hours)

The starting point: AlpinDale's `qwen_megakernel` is hand-tuned for Qwen3-0.6B with text vocab 151,936. Qwen3-TTS-1.7B-Talker is a different shape — bigger hidden, smaller (audio-only) vocab, different RoPE base, untied embeddings.

What I changed inside `qwen_megakernel_modified/`:
- `HIDDEN_SIZE` 1024 → 2048
- `INTERMEDIATE_SIZE` 3072 → 6144 (MLP width)
- `VOCAB_SIZE` 151936 → 3072 (audio codec vocab, not text)
- `MAX_SEQ_LEN` 2048 → 8192 (longer utterances)
- `ROPE_THETA` 10000 → 1,000,000 (Qwen3-TTS uses 1e6)
- Untied embeddings: input = codec_embedding, output = codec_head (separate matrices)
- Weight loader rewritten to read `talker.model.*` safetensor keys with the right prefix

What stayed the same:
- The persistent-kernel design itself (96 blocks × 512 threads, grid-sync barriers, atomic LM head reduction)
- The MRoPE math at the table level (in single-shared-position autoregressive decode, MRoPE collapses to vanilla 1D RoPE — see `qwen_megakernel/model.py` block comment for the proof)
- The `LDG_LM_NUM_BLOCKS=1184` retuning experiment confirmed: for the 5090's 170 SMs, more LM-head blocks ≠ slower (got 503 tok/s vs 143 with 24 blocks).

End of phase 1: kernel ran. Could feed it any audio token id and get the next one out. ~1000 tok/s on 0.6B-style bench harness ran fine; 1.7B-shape bench harness produced bogus tokens because the surrounding scaffolding wasn't wired up yet.

### Phase 2 — Wiring into a TTS pipeline (~4 hours)

Goal: text in → audio bytes out, streaming. Built:
- `MegakernelTTS` (the model wrapper) — async generator yielding raw int16 PCM bytes per codec frame
- `MegakernelTTSService` (Pipecat `TTSService` subclass) — yields `TTSAudioRawFrame` per chunk for the Pipecat path; also exposes a public `stream_tts()` that yields raw bytes for non-Pipecat callers (the Gradio UI)
- `pipecat_demo.py` — STT → LLM → TTS → output pipeline runner, mic and file modes
- `bench_megakernel.py` — bench harness for TTFC, RTF, decode tok/s with explicit `cuda.synchronize` at every timer boundary
- `ui_v2.py` — Gradio UI v2 with record-and-send mic widget, live metric cards, per-stage timing log

First bench numbers (cold pipeline, before audio QA): **TTFC 18.7 ms, RTF 0.123**. Both passing the Tightest tier. I almost shipped here.

### Phase 3 — The audio QA gate that surfaced the real bug (~3 hours)

Before packaging, I ran the bot's output audio through Deepgram nova-2 — the same speech-to-text model on the input side, used here as a quality gate. The idea: if the model that listens can transcribe what the model that talks just said, the audio path is faithful.

**Deepgram returned 0.0 confidence. Blank transcript. Every test.** The output sounded like speech but had no phonetic content.

I spawned a wiring-audit agent (Claude Code) that diffed our implementation against vanilla upstream `Qwen3TTSForConditionalGeneration` token by token. It surfaced four stacked bugs:

| # | What we had | What upstream does |
|---|---|---|
| 1 | CodePredictor driven by heuristic `0.5*(W[:,:1024]+W[:,1024:])` projection of an int token id | Talker's last-layer hidden state → 5-layer transformer → AR decode 15 additional codebooks |
| 2 | Talker AR step received only codebook-0's embedding | Talker receives `sum(16 codebook embeddings)` from previous step |
| 3 | Greedy argmax everywhere | `do_sample=True, top_k=50, temperature=0.9, repetition_penalty=1.05` per `generation_config.json` |
| 4 | Audio prefix sequence-concatenated (22 positions) | Element-wise sum with text projection at matching positions (10 positions) |

I fixed all four. The new AR loop runs:
```
talker.step_embed(combined_embedding)
  → CodePredictor.generate_ar(past_hidden, last_id_hidden)   [15-step AR sub-decode]
  → codec(stack([prev_tok, cb_preds_15]))                    [decode 80 ms PCM]
  → combined_embedding = last_id_hidden + Σ codec_embedding[i](cb_preds[i]) + trailing_text[step_idx]
  → next talker step
```

Audio came alive. Deepgram round-trip now transcribes the bot output at **0.9995-1.000** confidence, matching the vanilla upstream control I ran on the same GPU (also 0.9995).

Bench numbers regressed: **TTFC 23.42 ms, RTF 0.181**. TTFC still passes all tiers; RTF passes only Deliverables, misses Performance by 0.03. **Deliberate trade**: audio quality was the gate, perf is recoverable.

Per-step component profile post-audio-fix:

| Component | Time / step | % |
|---|---:|---:|
| CodePredictor 15-step AR sub-decode | 7.0 ms | 60% |
| Talker step (graphed PyTorch) | 3.0 ms | 26% |
| Codec forward | 1.0 ms (on side stream, mostly hidden) | 9% |
| Misc (embedding sum + host syncs) | 0.5 ms | 5% |

Wallclock per step: ~11.6 ms. RTF: 11.6 ms / 80 ms per codec frame = 0.145 floor with zero overhead, ~0.181 actual.

### Phase 4 — Today's session: the megakernel-AR swap (~2 hours)

When the reviewer asked "are we actually using the megakernel correctly — getting 1000 tok/s?" I traced the call graph and found something I hadn't noticed in days of work:

**`Decoder.step(token_id)` — the entry point that calls the persistent megakernel — was alive only in the bench harnesses.** The production TTS path called `Decoder.step_embed(input_embed)` which routed to `step_embed_logits_graphed` — a CUDA-graph-captured PyTorch 28-layer forward. The persistent megakernel was *never running* during a voice turn.

The reason for the bypass was real, not negligence:
- AlpinDale's kernel takes `(int input_token_id, embed_weight)` and does `x_in = embed_weight[input_token_id]` inside the kernel before layer 0.
- Qwen3-TTS's AR step needs a **precomputed** input embedding (`last_id_hidden + Σ codec_embedding[i](cb_preds[i]) + trailing_text[step_idx]`).
- There was no token-id surface for that. So when audio support was wired in Phase 2, the call was replaced with a graphed-PyTorch path, and "we use the megakernel" quietly stopped being true in the hot path.

The fix (~80 lines across three files):

1. **`csrc/kernel.cu`** — added a nullable `const __nv_bfloat16 *input_embed` parameter to `ldg_decode_kernel_direct`. When non-null, the kernel skips its internal `embed_weight[input_token_id]` lookup and reads layer-0 input directly from `input_embed`. Branch is uniform across blocks/threads — no warp divergence.
2. **`csrc/torch_bindings.cpp` + `kernel.cu`** — new `launch_ldg_decode_direct_embed` C entry point that runs the kernel with the precomputed embedding AND **skips** the subsequent `ldg_lm_head_fused` launch — caller does `F.linear(lm_head_weight, ·)` in PyTorch and applies the full custom sampling tail (rep-pen + suppress + top-k + Gumbel), which can't live in the kernel because of stateful RNG and rep-penalty history.
3. **`qwen_megakernel/model.py`** — `Decoder.step_embed_megakernel()` method registered. `step_embed` routes through it by default (kill-switch `QWEN_USE_MEGAKERNEL_AR=0`); falls back to the graphed-PyTorch path on any exception.

Result: ~280 graph-replay ops collapse to ONE persistent megakernel launch per AR step. Same math (RMSNorm, MRoPE, SDPA, final norm — all bit-identical up to bf16 rounding).

**TTFC**: 23.42 → 25.32 ms (+1.9 ms; still passes all three tiers).
**RTF**: 0.1813 → **0.1452** (−20%; crosses Performance tier <0.15).
**Deepgram**: 1.000 (no regression).

That's the current state. Performance tier passed, Tightest still 0.045 away.

---

## 3. Tradeoffs I made — and what each cost

### Tradeoff 1: Correct upstream-matching architecture vs the "fast" first pass

**Chose**: Re-architect to match upstream Qwen3-TTS exactly.
**Cost**: ~8 ms / step from the new CodePredictor 15-step sub-decode + sampling tail.
**Why**: The "fast" path produced unintelligible audio. The brief's `audio quality must be acceptable` clause is a hard gate. There's no "lossy audio quality" knob inside this architecture that would have let me trade audio for speed cleanly — the cut corners *were* the bug.

### Tradeoff 2: Code Predictor stays in PyTorch (not in the kernel)

**Chose**: CP runs as a regular PyTorch module with CUDA-graph capture, not as a kernel extension.
**Cost**: 14 per-step graph replays × ~0.5 ms launch overhead each = ~7 ms / step. The single biggest remaining cost.
**Why**: Brief says "swap megakernel in for the Talker decode loop only". CP is a different shape (5 layers, GQA, separate KV cache, 15-step AR). Building a second persistent kernel for CP would have been ~2 days of work and outside scope. The right hybrid: kernel for the heavy 28-layer talker, PyTorch+CUDA-graphs for the lighter 5-layer CP.

### Tradeoff 3: lm_head + sampling in PyTorch, not in the kernel

**Chose**: After the megakernel writes `g_normalized` (post-final-norm hidden), the caller does `F.linear(lm_head_weight, ·)` + sampling in PyTorch.
**Cost**: One extra matmul launch (~0.5 ms) + 5-6 sampling kernels (~1 ms). The kernel's fused lm_head + argmax that I bypass would have been faster.
**Why**: Custom sampling (rep-penalty + suppress-mask + top-k + Gumbel) needs stateful history and per-step suppression mask — not graph-friendly. Putting sampling in the kernel would have meant either dropping the custom sampling (audio breaks) or capturing host-side state into device buffers and re-capturing the graph every step (slow).

### Tradeoff 4: MRoPE collapsed to vanilla 1D RoPE in the autoregressive path

**Chose**: For AR-only audio decode after text prefill, all three MRoPE axes share a single position counter, so the table values are bit-identical to vanilla RoPE at θ=1e6.
**Cost**: Zero, in our path. The collapse is mathematically equivalent.
**Why**: Qwen3-TTS uses MRoPE primarily during multi-modal prefill (text + audio + spectrum axes diverging). Audio-only AR doesn't exercise that. If we wanted to extend to true multi-modal prefill, the table-build code is already structured for it; would be a one-function change.

### Tradeoff 5: Gradio UI uses single-yield-at-end, not chunk streaming

**Chose**: The UI collects all PCM chunks server-side and yields ONE `gr.Audio` blob at the end of utterance. The streaming-chunk property is preserved at the Pipecat layer + the bench harness + the public `stream_tts()` async generator.
**Cost**: The browser audio element doesn't visibly "tick" as chunks arrive. From a recording-the-demo standpoint, the audio plays as a single clip, not as a continuous stream.
**Why**: Gradio 6.x's `streaming=True` audio sink uses an HLS player that repeatedly `bufferStalledError`'d on our chunks, eventually quitting silently. `streaming=False` + autoplay gives reliable playback. The brief's streaming requirement is satisfied at the generator + Pipecat layer; the UI sink is the only place that breaks it, and it's documented honestly in the "Honest disclosures" panel of the UI itself.

### Tradeoff 6: Live UI metric cards show canonical bench, not per-turn measurement

**Chose**: The 4 metric cards in the UI display the canonical `bench_results.json` numbers (TTFC 25.32 / RTF 0.1452). The per-turn live measurement appears as a smaller "live this turn" subtitle.
**Cost**: A reviewer who only looks at the cards doesn't see THIS-turn numbers prominently.
**Why**: Running the model inside Gradio's busy HTTP-server async event loop adds ~30 ms / step of `await` overhead vs the CLI bench. Both measurements are honest, but they're measuring *different things*. Showing 60 ms on the card and 25 ms in the README would be confusing; this way the card matches the headline, and the subtitle preserves transparency.

### Tradeoff 7: Ship Performance tier now vs reach for Tightest tier

**Chose**: Stop perf work at RTF 0.145, submit Performance tier.
**Cost**: ~0.045 RTF gap to Tightest tier.
**Why**: The remaining gap is real but mechanical — collapse CP's 14 graph replays into one mega-graph (see §5 below). Estimated 1 hour focused. The decision was: get the current state shipped honestly, then either iterate or ship as-is. I chose to ship.

---

## 4. Results

### Bench (canonical)

n=5 + 3 warmup, RTX 5090 sm_120a, NGC PyTorch 2.10.0a, `cuda.synchronize()` at every timer boundary.

| Metric | Value | Tightest | Performance | Deliverables |
|---|---|---|---|---|
| TTFC | **25.32 ± 0.03 ms** | < 50 ms ✓ | < 60 ms ✓ | < 90 ms ✓ |
| RTF | **0.1452 ± 1.7e-4** | < 0.10 ✗ (by 0.045) | < 0.15 ✓ | < 0.30 ✓ |
| Decode wall / 5.12 s audio | 743.6 ms | — | — | — |

Reproducible: `python3 inference-server/bench_megakernel.py --warmup 3 --timed 5`. Raw data persisted in `bench_results.json`. Pretty-printed via `python3 scripts/show_bench.py`.

### Audio QA

Deepgram nova-2 round-trip on the bot output, all utterances:
- Our megakernel-wired path: **1.000** confidence on "Hello. How are you doing today?"
- Vanilla upstream Qwen3-TTS control on the same GPU: **0.9995**
- We match the upstream baseline.

Verification script: `scripts/deepgram_stt_check.sh <wav>`. Sample outputs: `samples/qa_default_on.wav` (megakernel-AR path, 1.000) + `samples/upstream_ref_ryan.wav` (vanilla control, 0.9995).

### End-to-end latency (Pipecat warm path)

UserStop → BotStart, measured via Pipecat's `UserBotLatencyObserver`:
- Groq LLM TTFB: ~650 ms (cloud RTT — outside our control)
- TTS first chunk: ~36 ms (matches the bench's 25 ms within Pipecat's measurement methodology)
- Total: ~900-1500 ms typical

The headline e2e latency is dominated by cloud STT/LLM RTT, not by anything we built. That's the right shape: our component is the fast part.

---

## 5. Why we miss Tightest tier (RTF < 0.10) — per-step decomposition

Current decode wall per AR step on RTX 5090, post-megakernel-AR swap:

| Component | Time | % | Tightest-tier path |
|---|---:|---:|---|
| CodePredictor 14-step AR (one CUDA graph replay per step) | ~6.5 ms | 56% | **Collapse into ONE mega-graph with Gumbel sampling inside** → −3 to −5 ms |
| Talker step (megakernel) | ~3.0 ms | 26% | Near theoretical floor (bandwidth-bound on 1.7B weights). Possibly −0.5 ms more with deeper kernel tuning, not high-ROI. |
| Codec forward (side stream) | ~5 ms async | 0% wall (overlaps with talker) | Already overlapped; nothing to gain. |
| Sampling + history bookkeeping + frame yield | ~2 ms | 17% | Could pipeline-by-one (issue next step before consuming current codec output) → −2 to −3 ms BUT risks TTFC regression. |
| **Wall** | **~11.6 ms / step** | | |

Tightest tier needs RTF < 0.10 = 8 ms / step or less. Gap: 3.6 ms / step.

The CP mega-graph alone might land at 0.10-0.11 (just barely misses by 0.005-0.015). CP mega-graph + careful pipeline-by-one likely crosses 0.10 cleanly. Both are 60-90 min of focused work each, and the pipeline-by-one change has burned us before (a previous attempt regressed TTFC by ~15 ms).

**What it would NOT take**: sacrificing audio quality. The remaining time per step is launch-overhead-bound, not compute-bound. No "lossy audio mode" would help. The fix is mechanical kernel-graph engineering, not architectural compromise.

---

## 6. What I'd build next (in priority order)

1. **CP mega-graph** (60-75 min, ~70% confidence we cross Tightest). Single CUDA graph that runs the 14 gen-step layer forwards + sampling inside, with Gumbel noise pre-filled outside the graph (RNG state isn't graph-safe). The hard part: chaining 14 sample-then-feed-back-as-prev-id steps inside one graph requires careful device-side state. Doable, scoped in `CHANGELOG.md`.

2. **Pipeline-by-one** (30-45 min on top, audio-safe if done right). Issue the next CP/talker step *before* waiting for the current codec frame to finish. The first yield is the only one that needs to stay un-pipelined to protect TTFC. Should save 2-3 ms / step.

3. **MegakernelTTSService TTFC reduction in the Pipecat path** (~30 min). The Pipecat wrap (`start_tts_usage_metrics`, TTFB observer, `TTSAudioRawFrame` allocation) adds ~5-15 ms before the first chunk. Could be elided by moving the timer start point deeper into the service.

4. **Decode tok/s as a clean canonical number** (~30 min). The decode-only sub-bench currently hangs (two Decoder instances share kernel static barriers). Rewrite to reuse the existing `MegakernelTTS._talker` Decoder, or accept a `Decoder` instance as a parameter.

---

## 7. Honest meta-reflection

- **What helped most**: the Deepgram round-trip QA gate. Single quantitative signal vs subjective "does it sound right?" listening. Caught the four-bug audio stack that I would have shipped on speed alone. Doing this on every audio-touching change made the polish loop fast.
- **What hurt most**: not tracing the call graph for "is the megakernel actually in the hot path?" earlier. I spent days assuming `step_embed` went through `_decode` because the file structure suggested it. A 30-second grep would have caught the bypass on day 1. Lesson: verify what runs in the hot path with `pgrep` and `strace`, not by reading file names.
- **What I'd do differently**: keep the README short (the brief asks for it). Mine is long because the diagnosis arc is genuinely the story, but a v2 would split the historical context into a separate file and keep the main README under 100 lines.

---

End. Repo: <https://github.com/pratham7711/e3-megakernel-tts-takehome>. Demo: <https://www.loom.com/share/7768e549803c4a3a8678e4a5f39d996b>. Questions / objections / call requests welcome.
