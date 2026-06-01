# Changelog

All notable changes to this take-home submission are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
does not version-tag (single submission), so everything lives under
`[Unreleased]`.

## [Unreleased]

### Added — Megakernel-AR talker swap (2026-06-01, Performance tier achieved)

**The earlier perf gap (RTF 0.181, missing Perf <0.15 by 0.03) had a load-bearing
diagnosis:** the production AR audio-decode hot path was NOT going through the
persistent megakernel. `Decoder.step(token_id)` (which calls
`torch.ops.qwen_megakernel_C.decode`) was alive only in the bench harnesses
(`bench.py`, `bench_megakernel.py`). The actual TTS path called
`Decoder.step_embed(input_embed)` which routed to a CUDA-graph-captured
PyTorch 28-layer forward (`step_embed_logits_graphed`), because the megakernel
took a token-id input and the Qwen3-TTS AR step needs a *precomputed* embedding
(`last_id_hidden + Σ codec_embedding[i](cb_preds[i]) + trailing_text[step_idx]`).

**Fix landed in this session** (kernel surgery, ~80 lines across
`csrc/kernel.cu`, `csrc/torch_bindings.cpp`, `qwen_megakernel/model.py`):

1. Added a nullable `input_embed: const __nv_bfloat16*` param to
   `ldg_decode_kernel_direct`. When non-null, the kernel skips the internal
   `embed_weight[input_token_id]` lookup and reads layer-0 input directly from
   `input_embed`. Branch is uniform across blocks/threads — no warp divergence,
   no measurable launch-overhead change.
2. Added `launch_ldg_decode_direct_embed` C entry point that calls the kernel
   with the precomputed embedding AND skips the subsequent `ldg_lm_head_fused`
   launch — caller does `F.linear(lm_head_weight, …)` in PyTorch and applies
   the full custom sampling tail (rep-pen + suppress-mask + top-k + Gumbel,
   which can't live inside the kernel).
3. Registered `torch.ops.qwen_megakernel_C.decode_embed` Python op +
   `Decoder.step_embed_megakernel()` method. `step_embed` routes through it
   by default (kill-switch: `QWEN_USE_MEGAKERNEL_AR=0`); falls back to the
   graphed-PyTorch path on any exception.

**Bench numbers (n=5, warmup=3, RTX 5090 sm_120a, same gates as the previous
canonical numbers):**

| Metric | Before (graphed PyTorch) | After (megakernel-AR) | Δ | Tier |
|---|---|---|---|---|
| TTFC | 23.42 ± 0.03 ms | **25.32 ± 0.03 ms** | +1.9 ms | ✅ all three tiers |
| RTF  | 0.1813 ± 2.8e-5  | **0.1452 ± 1.7e-4**   | −20% | ✅ **Performance <0.15** |
| Decode wall (5.12 s audio) | 928 ms | **743 ms** | −185 ms | — |

**Audio QA gate (Deepgram nova-2 round-trip)**: "Hello. How are you doing today?"
→ "Hello. How are you doing today?" at **confidence 1.000**. Matches the upstream
control + pre-swap baseline (0.9995).

**Math equivalence**: same RMSNorm, same MRoPE table values (single-shared-position
1D collapse), same SDPA, same final norm. Only the dispatch shape changes —
~280 graph-replay ops collapse to ONE persistent megakernel launch (96 blocks ×
512 threads, `LDG_LM_NUM_BLOCKS=1184` for the 5090's 170 SMs). lm_head + sampling
remain in PyTorch.

**The remaining perf gap to Tightest tier** (RTF <0.10, need −0.045 more):
the next lever is collapsing CP's 14 per-step CUDA-graph replays + Gumbel
sampling into ONE megagraph (estimated 3–5 ms/step, ~0.13 RTF). Scoped but
not implemented this session.

### Fixed — Megakernel ↔ Qwen3-TTS wiring (2026-06-01, audio gate finally passes)

The earlier "Audio intelligibility fix" commit (`18f2123`) corrected one of four
stacked wiring bugs but left the other three; Deepgram STT round-trip returned
blank transcripts on the generated audio at 0.0 confidence across every speaker
and input variant. A spawned wiring-audit agent diagnosed the full four-bug
stack and a follow-up implementation agent closed all four.

**Bugs fixed**:
- **C1 — CodePredictor input was a heuristic int-token lookup.** Upstream feeds
  the talker's last-layer hidden state plus the prev codec token's embedding
  through a 5-layer transformer that AR-decodes 15 additional codebooks. Our
  previous path collapsed this into a single-step parallel argmax on a
  heuristically-projected (`0.5*(W[:,:1024]+W[:,1024:])`) input. Fix: real
  15-step AR sub-decode in `CodePredictor.generate_ar` with per-codebook
  embedding/lm_head tables loaded from `talker.code_predictor.*` safetensors.
- **C2 — Talker AR step was fed codebook-0 alone.** Upstream feeds
  `sum(16 codebook embeddings)` from the previous step. Fix: AR loop rewrite in
  `MegakernelTTS.generate()` that consumes CodePredictor output and sums the 16
  embeddings before the next `step_embed` call.
- **C3 — Greedy argmax on a sampling-trained model.** Upstream
  `generation_config.json` is `do_sample=True, top_k=50, temperature=0.9,
  repetition_penalty=1.05`. Fix: sampling in both `Decoder._sample_audio_token`
  (talker) and `CodePredictor._sample_logits` (per-codebook). Tightened to
  `temp=0.5, top_k=20` after observing inter-phoneme outlier jumps at
  `temp=0.9` — confirmed via Deepgram round-trip that the tighter distribution
  preserves intelligibility while removing distortions.
- **C4 — Audio prefix was sequence-concatenated.** Upstream builds the prefix as
  an element-wise sum of text projections and codec embeddings at matching
  positions (10 positions: 3 role + 6 merged + 1 first_text+codec_bos). Fix:
  rewrite `Decoder.prefill_text` to match upstream
  `modeling_qwen3_tts.py:2174-2202` exactly, drop the stray
  `<tts_text_bos>/<tts_text_eod>` ids.

**Audio QA gate** (Deepgram nova-2 round-trip):
- ryan + "Hello. How are you?" → "Hello. How are you?" / 0.9995
- aiden + "Hello. How are you?" → "Hello. How are you?" / 1.000
- ryan + "Hello." → "Hello?" / 0.960
- aiden + "Hello." → "Hello?" / 0.979
- ryan + long sentence ("Hello, this is a test of the megakernel speech synthesis system. The quick brown fox jumps over the lazy dog. How are you today?") → full transcript / 0.9990

Control: same input through vanilla `qwen_tts.Qwen3TTSModel` (no megakernel) on
the same GPU produces the same Deepgram transcript at 0.9995 confidence.
Wiring is faithful.

**Perf regression — deliberate**: the four fixes added a 15-step CodePredictor
AR sub-decode per AR step (eager PyTorch). Bench numbers moved from
TTFC 18.7 ms / RTF 0.123 (passing perf+deliverables tiers, missing tightest by
23%) to **TTFC 23.42 ms / RTF 0.181** (after team perf+audio polish). TTFC passes ALL
three brief tiers (Tightest <50 / Perf <60 / Deliverables <90 ✅✅✅); RTF
passes Deliverables (<0.30 ✅), misses Perf (<0.15) by 0.03, misses Tightest
(<0.10) by 0.08. Landed across two parallel agents: CUDA-graph capture on
talker `step_embed` (Move A) and CodePredictor inner loop (Move C); KV-cache
hoist + `enable_gqa=True` SDPA (Move B); Gumbel-max sampling (Move E);
stacked `F.embedding` gather for the 16-codebook embedding sum (Move F);
`torch.compile(reduce-overhead)` re-enabled on per-frame codec (Move D);
host-sync reduction (Move E2); pre-allocated per-step scratch tensors. Audio
polish: sampling temperature retuned from 0.5/20 to 0.7/30 (both talker and
CodePredictor) — moved all 4 spectral metrics decisively toward the upstream
baseline (F2/F1 ratio 2.14 vs upstream 2.15; RMS 0.131 vs upstream 0.127),
fixing the user-reported "voice distortion" without breaking Deepgram.

The perf gap is well-scoped: CodePredictor.generate_ar consumes 76.9% of every
AR step per the per-component profile (`profile_results.json`); known fixes
(remaining CUDA-graph capture optimizations, kernel-side `step_embed` accepting
a precomputed input embedding, full-AR-loop torch.compile) project the AR step
to 35-75 ms, putting RTF in the 0.20-0.45 range with no further audio-quality
change.

### Changed

- Sampling defaults — both talker and CodePredictor — locked at
  `temperature=0.5, top_k=20, repetition_penalty=1.05` after polish-pass A/B
  testing (`model.py:_sample_audio_token`,
  `qwen3_tts_components.py:_sample_logits`).
- `MegakernelTTSService.stream_tts(text, *, max_new_tokens)` public wrapper
  replaces the prior `_tts.generate(...)` reach-through used by `ui_v2.py`.
- `ui_v2.py` requests browser mic + speaker permission on `DOMContentLoaded`
  (Gradio 6 `launch(head=...)`) — green/red status pill renders at the top of
  the dashboard; mic-grant on Chromium/Firefox also unlocks `<audio>` autoplay
  so TTS streaming starts without an explicit "play" click.
- New `scripts/upstream_ref_test.py` — runs vanilla `qwen_tts.Qwen3TTSModel`
  on the same GPU + weights as a ground-truth Deepgram-QA baseline.
- New `scripts/deepgram_stt_check.sh` — the QA gate that the audio-fix
  iteration loop uses.
- New `scripts/run_voice_turn.sh` — Mac mic → GPU pipeline → Mac speaker
  round-trip script for demo recording.
- New `inference-server/generate_test_audio.py` — single-utterance TTS-only
  generator for Deepgram QA (no Pipecat orchestration).

### Added

- **Streaming text-prefill into the megakernel KV cache.** New
  `Decoder.prefill_text(text)` runs a pure-PyTorch 28-layer forward and writes
  K/V directly into the megakernel's `_k_cache` / `_v_cache` slots, so
  autoregressive decode is conditioned on the input string without requiring
  a kernel-side prefill path. Numerical drift bounded by fp32-vs-fused-bf16
  differences. (`501f6ff`)
- **Per-frame streaming yield** in `MegakernelTTS.generate()`. Caller receives
  the first 1920-sample PCM frame within ~17 ms of the first talker token
  (Config A, sine-stub codec); previous build flushed only at end-of-utterance.
  (`501f6ff`)
- **Polished Gradio UI v2** (`inference-server/ui_v2.py`): dark-themed Blocks
  dashboard with metric cards (TTFC, RTF, tok/s, audio duration) color-coded
  against the brief's three target tiers, comparison table, run history (last
  10 runs), build-flags sidebar, and explicit honest-disclosures section.
  (`64370fe`)
- **End-to-end bench harness** (`inference-server/bench_megakernel.py`) that
  measures all four brief metrics (TTFC, RTF, decode tok/s, audio duration)
  with `n=5` + 3 warmup, writes `bench_results.json`. (`64370fe`)
- **Real Qwen3-TTS V2 codec.** Clean-room reimplementation in
  `inference-server/qwen3_tts_components.py`; all 271 codec weights load
  with 0 missing / 0 unexpected. Output is broadband voiced/unvoiced
  spectrum (`demo_audio_real_codec.wav`). Replaces the prior sine-wave
  stub. (`0f457c9`)
- **Pipecat skeleton end-to-end.** STT (Deepgram) -> LLM (Groq
  llama-3.1-8b-instant) -> our `MegakernelTTSService` -> `LocalAudioOutputTransport`,
  with Silero VAD and `LLMContextAggregatorPair` wired per Pipecat
  conventions. (`62187ee`, `501f6ff`)
- **`INPUT_MODE=mic|file` switch** in `pipecat_demo.py` with custom
  `WavFileInputProcessor` source + `AudioBufferProcessor` sink, so the demo
  runs headless on a GPU box without PyAudio. (`501f6ff`)
- **MRoPE cos/sin table builder** (`_build_mrope_tables` in
  `qwen_megakernel_modified/qwen_megakernel/model.py`) under
  `rope_theta = 1,000,000` with mrope_section semantics; collapses to vanilla
  1D RoPE for the autoregressive-only path (verified bitwise). (`62187ee`)
- **KV cache correctness suite.** 5/5 checks pass: deterministic across
  resets, monotonic positions, no out-of-range tokens, prompt-conditioned
  outputs, `reset()` actually clears. (`64370fe`)
- **Groq + HF_TOKEN env-vars** added to `.env.example`. (`7ea4808`)

### Changed

- **LM-head block tuning reverted to `LDG_LM_NUM_BLOCKS = 1184`.** Intuition
  said shrink the block count when the vocab shrinks 50x; measurement said
  the opposite. The RTX 5090 has 170 SMs and even small-vocab LM-head is
  bandwidth-bound -- high block count keeps occupancy up. 24 blocks gave
  143 tok/s; 1184 blocks give **503 tok/s**. Documented at the top of
  `qwen_megakernel_modified/qwen_megakernel/build.py`. (`64370fe`)
- **Code predictor + codec batching.** `ui_v2.py` now runs the talker step
  in a tight loop collecting all semantic tokens, then makes one
  `code_predictor((1, N, 16))` call and one `codec((1, N*1920))` call
  per utterance (was calling codec 25 times per utterance for the streaming
  path). Cut first-run generate from 10+ minutes to ~1.25 s. (`a625fa9`)
- **Codec int16 quantisation** made explicit: codec output is float
  `[-1, 1]`, the UI now quantises to int16 before WAV write. (`a625fa9`)
- **Pipecat LLM default** switched to Groq (matches `.env`). Verified
  `GroqLLMService.Settings` pattern against upstream source. (`501f6ff`)
- **Sample rates aligned**: STT 16 kHz, TTS 24 kHz, `AudioBufferProcessor`
  24 kHz so the recorded WAV doesn't get resampled. (`501f6ff`)
- **README rewrite** with the honest A/B split between Config A (sine-stub
  codec, isolates megakernel + code_predictor) and Config B (real 271-weight
  codec, cold-start dominated). Mermaid flowchart at top, 7-step "How to run"
  walkable on a fresh Vast box, decisions log, "what I'd do with another
  day" expanded to 8 items. (`9307ae0`, `b64dbb8`, `501f6ff`)

### Fixed

- **(HIGH) Silent codec stub fallback.** `megakernel_tts.py:222`
  `load_components()` returns a 3-tuple `(cp, codec, info)` since the
  real-codec rewrite; the caller was unpacking 2 entries, silently flipping
  the service to STUB mode emitting silence. Restored to real-codec path.
  (`501f6ff`)
- **(HIGH) Bench harness signature mismatch.** `bench_megakernel.py:169` --
  `Decoder` takes `model_path` not `model_name`, and the new `model.py`
  doesn't expose `.tokenizer` (safetensors load skips HF tokenizer). Bench
  now seeds with token id 0; talker-only bench measures decode throughput
  cleanly. (`501f6ff`)
- **(HIGH) TTFC misreported.** `ui_v2.py:295` was measuring TTFC at
  full-utterance batched flush; now correctly measured at first frame's
  emission (post-`cuda.synchronize`). (`501f6ff`)
- **(HIGH) Codec sliding-window mask `O(T^2)`.**
  `qwen3_tts_components.py:_sliding_mask` was allocating a dense `(T, T)`
  float `-inf` tensor on every forward, blocking SDPA's fast path. Returning
  a bool mask (~1 KB) lets SDPA short-circuit. Cut second-utterance hang
  from ~80 s to ~0.5 s. (`501f6ff`)
- **int-vs-tensor type mismatch** at `megakernel_tts.py:297` -- was passing
  a Python int into `code_predictor`; now wraps as `(1, 1)` `LongTensor`.
  (`64370fe`)
- **UI unpacking + Gradio kwargs.** `prepend_history()` crashed on
  `history or []` when Gradio Dataframe passed a `pandas.DataFrame` (ambiguous
  truthiness); routed via `.empty` instead. Removed unsupported Gradio
  kwargs (`show_download_button`, `show_api`). (`0f457c9`)

### Security

- `.env` is now `.gitignore`d to prevent API key leakage; only
  `.env.example` is tracked. (`7ea4808`)

---

## [0.1.0] -- 2026-05-30 (Initial submission, `29fbac3`)

The starting state of this repo, before the multi-sprint polish run above.

### Added

- AlpinDale `qwen_megakernel` ported for the Qwen3-TTS-1.7B talker:
  - `HIDDEN_SIZE` 1024 -> 2048, `INTERMEDIATE_SIZE` 3072 -> 6144
  - `LDG_VOCAB_SIZE` 151936 -> 3072 (audio codebook)
  - `MAX_SEQ_LEN` 2048 -> 8192, `rope_theta` 10000 -> 1,000,000
  - Untied embeddings: `codec_embedding` (3072 x 2048) in,
    `codec_head` (3072 x 2048) out
  - `model.py` weight loader rewritten for `talker.model.*` safetensors keys
- Initial Pipecat `TTSService` skeleton + bench harness + demo wiring.
- Performance baseline: **503.1 tok/s, 1.988 ms/tok** on the modified 1.7B
  talker, vs 1034.6 tok/s on the stock 0.6B baseline (2x slower; predicted
  3x from weight scaling; LM-head 50x shrink offsets the rest).

### Known Limitations

- MRoPE multi-axis math not implemented inside the CUDA kernel; the cos/sin
  tables collapse to vanilla 1D RoPE @ `theta=1M`. Mathematically exact for
  the pure autoregressive decode loop; diverges from HF reference during
  true multi-axis prefill (see `ARCHITECTURE.md` Section 4).
- Codec was a sine-wave stub; replaced in the polish run above.

[Unreleased]: https://github.com/pratham/e3-megakernel-tts/compare/29fbac3...HEAD
[0.1.0]: https://github.com/pratham/e3-megakernel-tts/commit/29fbac3
