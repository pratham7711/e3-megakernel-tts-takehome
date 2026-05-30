# Changelog

All notable changes to this take-home submission are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
does not version-tag (single submission), so everything lives under
`[Unreleased]`.

## [Unreleased]

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
