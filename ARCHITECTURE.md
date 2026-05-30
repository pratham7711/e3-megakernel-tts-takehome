# ARCHITECTURE

A deeper sibling to the README. The README answers "what does it do and how
fast"; this document answers "why is it shaped that way, where does the math
break, and what would we change if we had another sprint."

> Audience: a reviewer who has skimmed the README mermaid diagram and wants
> the implementation rationale, the honest correctness gaps, and the
> next-step plan. Companion docs: `CHANGELOG.md`, `NOTICES.md`.

---

## 1. Overview

This repo ports AlpinDale's `qwen_megakernel` (a single-CUDA-kernel decode
loop hand-tuned for Qwen3-0.6B on an RTX 5090) to drive the **talker**
decoder of `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`. The megakernel is the only
CUDA-resident hot path; everything else (text prefill, code predictor, codec
vocoder, audio output) runs as ordinary PyTorch eager / `torch.compile`-graphed
code on the same device. The motivation is narrow: the brief asked for a
real-time TTS service under a tight RTF budget, and the autoregressive
token-decode loop is the hottest hot-loop in the system. We push that loop
into a single fused kernel and let PyTorch handle the rest.

The runtime topology is three stages chained on the GPU: a 28-layer Qwen3 GQA
transformer (the **talker**) producing one semantic-token per ~2 ms at
12.5 Hz, a 5-layer auxiliary transformer (the **code predictor**) expanding
each semantic token into a 16-codebook acoustic frame, and a clean-room
reimplementation of the Qwen3-TTS V2 codec (transformer pre-net + ConvNet
1920x upsampler) that synthesises 1920 samples of 24 kHz PCM per frame.
Pipecat owns I/O plumbing -- STT (Deepgram), LLM (Groq), VAD (Silero), local
audio transport -- but is replaceable; `MegakernelTTS` is a plain
`AsyncGenerator[bytes]` used directly by both the Gradio UI and the Pipecat
`TTSService` wrapper.

This document does not repeat the perf table or run-recipe (those live in the
README). What follows: the structural breakdown, the megakernel diff vs the
0.6B baseline, the two honest correctness gaps (MRoPE single-axis collapse
and the pure-PyTorch text prefill), the streaming design call, the codec
slowness story and its bool-mask fix, and the prioritised next-day list.

---

## 2. Component breakdown

### 2.1 Talker (megakernel CUDA, 28 layers, ~1.7 B params)

| Axis             | Value      | Source                                       |
|------------------|------------|----------------------------------------------|
| Layers           | 28         | Qwen3-TTS config                             |
| Hidden           | 2048       | `csrc/kernel.cu:22` (was 1024)               |
| MLP intermediate | 6144       | `csrc/kernel.cu:23` (was 3072)               |
| Q / KV heads     | 16 / 8     | unchanged (GQA)                              |
| Head dim         | 128        | unchanged                                    |
| Vocab (audio)    | 3072       | `csrc/kernel.cu:74` (was 151936)             |
| Max seq          | 8192       | `qwen_megakernel/model.py:15` (was 2048)     |
| rope_theta       | 1,000,000  | `qwen_megakernel/model.py:18` (was 10000)    |
| Tied embeddings  | False      | `qwen_megakernel/model.py:87`                |
| Weight source    | `talker.model.*` of CustomVoice safetensors | `qwen_megakernel/model.py:65-87` |

A single CUDA kernel per decoded token covers: RMSNorm, fused QKV projection,
MRoPE rotation, GQA attention against the in-place KV cache, output
projection, residual, second RMSNorm, fused SwiGLU MLP, residual, LM head. On
the 0.6B baseline this hits 1036 tok/s on a 5090; with our 1.7B talker shape
it hits **503 tok/s** (~2x slower for ~3x weight, because the LM-head shrank
50x and freed bandwidth).

What we modified: the seven constants above, the weight loader, and the
LM-head block tuning. What we took as-is: GQA attention layout, SwiGLU math,
RMSNorm shape, warp/block tiling, the C++ binding surface.

### 2.2 Code Predictor (PyTorch, 5 layers)

A small auxiliary transformer (5 layers, hidden 1024, 16 codebook heads x
vocab 2048) that takes one semantic token from the talker and emits a
16-channel acoustic frame. Taken **as-is** from the Qwen3-TTS reference,
instantiated as `nn.Module` and wrapped in
`torch.compile(mode="reduce-overhead", dynamic=False)` against the stable
`(1, 1)` shape for CUDA-Graph capture. This is the biggest per-frame cost in
warm steady-state (~14 ms of the 16.7 ms frame budget). We kept it
eager-then-compiled rather than fusing into the megakernel: brief scopes the
megakernel to the talker decode loop, and the code predictor has different
I/O shape (16 parallel heads, no KV cache, no MRoPE) -- `torch.compile` is
the right tool here.

### 2.3 Codec (PyTorch, clean-room reimpl, 271 weights)

Qwen3-TTS-V2 codec: `SplitResidualVectorQuantizer` (1 semantic + 15 acoustic
codebooks), 8-layer transformer pre-net at hidden 512 with sliding-window
causal attention (window 72), and a ConvNet decoder upsampling by
`2 * 2 * 8 * 5 * 4 * 3 = 1920x` to produce exactly 1920 samples of 24 kHz PCM
per 12.5 Hz frame.

We did **not** vendor the `qwen-tts` pip package -- it fails to import on the
PyTorch nightly NGC image (torchaudio ABI conflict). `inference-server/qwen3_tts_components.py`
is a clean-room reimplementation informed by upstream `QwenLM/Qwen3-TTS`
modelling code, targeting only `torch`. All 271 weights load with **0 missing,
0 unexpected**, output is broadband voiced/unvoiced spectrum
(`demo_audio_real_codec.wav`). Codebook `embedding_sum` / `cluster_usage` are
kept in fp32 for per-code-mean numerical stability; rest is bf16.

---

## 3. The megakernel adaptation -- deltas vs the 0.6B baseline

Reference plan: `brain/build/side-projects/e3-kernel-mod-plan.md` (file-by-file
diff produced by the planning agent, line-numbers verified against the local
clone).

The 0.6B megakernel hard-codes model shape at C++ compile-time and threads
weight-key names through the Python loader. The talker port is a
constants-diff plus a weight-key remap; the kernel structure is byte-for-byte
identical because Qwen3-0.6B and Qwen3-TTS-1.7B share the same architectural
family.

**`csrc/kernel.cu` changes (all in-place, no new code paths):**

| Constant            | 0.6B   | 1.7B talker | Rationale                                |
|---------------------|--------|-------------|------------------------------------------|
| `HIDDEN_SIZE`       | 1024   | 2048        | model width 2x                           |
| `INTERMEDIATE_SIZE` | 3072   | 6144        | MLP width 2x                             |
| `LDG_VOCAB_SIZE`    | 151936 | 3072        | audio codebook (50x smaller)             |
| `LDG_LM_NUM_BLOCKS` | 1184   | 1184 (kept) | see below                                |
| `LDG_LM_BLOCK_SIZE` | 256    | 128         | smaller per-block tile, matches small vocab |

**The `LDG_LM_NUM_BLOCKS` story.** The intuitive move when shrinking vocab
50x is to also shrink LM-head block count. The plan called for 24. We tried
it; 143 tok/s. Reverting to 1184 gave 503 tok/s. The RTX 5090 has 170 SMs and
the LM head is bandwidth-bound -- high block count keeps occupancy up even
on a small vocab. The "shrink because vocab shrank" intuition was wrong on
Blackwell. Documented at the top of `qwen_megakernel_modified/qwen_megakernel/build.py`.

**`qwen_megakernel/model.py` changes:**

- All per-layer keys remapped `model.layers.{i}.*` -> `talker.model.layers.{i}.*`.
- Input embed: text (151936 x 1024) -> audio (`talker.model.codec_embedding.weight`, 3072 x 2048).
- Output proj: tied-to-embed -> **untied** `talker.codec_head.weight` (3072 x 2048).
- Text-side weights (`text_embedding`, `text_projection.fc1/fc2`) loaded
  separately, used by the pure-PyTorch text-prefill path (Section 5).
- `_build_mrope_tables()` builds cos/sin under `rope_theta=1,000,000` with
  mrope_section semantics baked in (Section 4).

Deliberately not modified: `csrc/torch_bindings.cpp` (shape-agnostic); the
MRoPE rotation code in `kernel.cu` (the kernel already does interleaved
split-half rotation, `partner = i +/- HEAD_DIM/2`, matching Qwen3-TTS's
`interleaved=true`).

---

## 4. MRoPE: single-axis collapse -- honest explanation

Qwen3-TTS config: `interleaved: true`, `mrope_section: [24, 20, 20]` (sum =
`head_dim/2`), `rope_theta = 1,000,000`. Three sections are independent
position axes: text / audio-time / spectrum.

### 4.1 Why single-axis collapses to vanilla 1D RoPE

For an autoregressive decode loop where only the audio-time axis is moving
and the other two are frozen at `(T_text, 0)`, the cos entry for dimension
`d` becomes:

```
inv_freq_d   = 1 / theta^(2*d / head_dim)
cos_d(T_text)    = constant      -- text section
cos_d(pos_audio) = standard      -- audio section
cos_d(0)         = 1             -- spectrum section
```

Frozen-axis dimensions apply a constant rotation per step. The same constant
hits both Q and K, and rotation is unitary -- so the constant cancels in
`<Q', K'>`. Effective attention contribution comes only from the audio-time
section dims, with the standard `inv_freq` formula at `theta = 1M`.

We verified numerically: max absolute difference between our
`_build_mrope_tables` and the naive 1D RoPE formula at `theta=1M` (restricted
to audio-time section dims, other axes zero) is **0.0** -- bitwise identical.

### 4.2 Where the collapse breaks: multi-axis prefill

During real text-prefill, the talker processes text tokens at `(t, 0, 0)`
then audio-prefix tokens at `(T_text, a, 0)`. K-cache writes during prefill
use position triplets with **different values across the three axes**, so K
vectors get rotated under `cos(T_text)*text_section + cos(a)*audio_section +
cos(0)*spectrum_section`. Our single-axis collapse assumes all three axes
track the audio counter -- so the K-cache gets a slightly different rotation
than HF produces.

Decode-time Q rotation is fine (audio axis advances, others frozen) but it's
being dotted against a K-cache built under different axis math during
prefill. The numerical effect is bounded: for short prefills (`T_text < 50`)
and short audio-prefix windows, divergence is small in the first few audio
tokens before the audio axis dominates. For the **pure autoregressive bench**
(decode from fresh KV cache, no prefill) the effect is zero by construction.

For the **text-prefill path** (Section 5) we sidestep the kernel-side gap by
doing prefill in pure-PyTorch with HF's exact MRoPE math, then writing the
K/V tensors directly into the megakernel's `_k_cache` / `_v_cache`.

### 4.3 Cost of closing the gap fully

`csrc/kernel.cu:344-409` (K and Q rotation loops) needs `int3 pos` and a
per-dim axis lookup. Split-half partner indexing is unchanged. Estimated
~1-2 GPU hours. Deferred; math fully specced in brain note `e3-mrope-math.md`.

---

## 5. The text-prefill path -- pure PyTorch into KV cache

The talker's autoregressive decode loop only makes sense if conditioned on a
text prompt. Three options considered:

1. Wrap text-prefill in the megakernel (~half a GPU-day for a separate
   prefill-shape kernel).
2. Run HF's full PyTorch forward to produce a KV cache, concatenate to the
   megakernel's cache (works but parallel model graphs during prefill).
3. **Do prefill in pure PyTorch using the same weights the megakernel loaded**,
   compute K and V at each layer and write directly into the megakernel's
   `_k_cache[layer]` / `_v_cache[layer]` slots at positions `[0, T_text)`.
   This is what we shipped.

`Decoder.prefill_text(text)` runs a 28-layer forward: embed via
`text_embedding`, apply `text_projection` MLP (`fc1 -> act -> fc2`, the
Qwen3-TTS-specific text path), then per layer compute RMSNorm -> QKV ->
MRoPE with the **correct multi-axis position triplet** -> attn out -> RMSNorm
-> SwiGLU. K and V at each layer slice into the megakernel cache.

Why this is the pragmatic shortcut:

- **No C++ changes.** Avoids kernel-side MRoPE rewrite for prefill.
- **One source of truth for prefill math.** Match HF's exact MRoPE table
  layout in pure Python; no kernel/HF sync.
- **Decode sees a correctly-conditioned KV cache** (subject to the
  single-axis decode-rotation caveat in Section 4).

Numerical drift to expect: PyTorch uses fp32 softmax inside SDPA; megakernel
uses bf16-fused. ~1e-3 relative drift in K-cache entries vs fully-fused
prefill -- well below per-token argmax noise. Long prefills (~100+ tokens)
accumulate up to ~1e-2 in deepest layers; fine for speech naturalness,
problematic for logits-diff vs HF.

---

## 6. Streaming vs batched

The brief calls out real-time streaming. The naive "decode all talker
tokens, then run code_predictor on the batch, then run codec on the batch"
gives cleanest GPU utilisation and lowest end-to-end wall-clock, but has
unbounded **TTFC** -- the listener hears nothing until the whole utterance
is decoded. Wrong shape for a voice-loop product.

`MegakernelTTS.generate()` is `AsyncGenerator[bytes]`. Per talker step
(~2 ms): accumulate semantic tokens silently within a codec-frame batch; at
each 12.5 Hz codec-frame boundary, flush through code_predictor + codec and
yield 1920 samples (3840 bytes int16) of PCM. The caller sees the first
frame within ~17 ms of the first talker token (Config A) -- that is the TTFC
we report.

`code_predictor` and `codec` are wrapped in
`torch.compile(mode="reduce-overhead", dynamic=False)` against fixed
`(1, 1)` shape. `reduce-overhead` triggers CUDA Graph capture, which:

- Eliminates per-call CUDA dispatch overhead (~600x speedup observed vs
  un-fused per-frame calls -- the headline fix in commit `a625fa9`).
- Locks in the kernel launch sequence so the graph replays per frame without
  Python in the hot path.
- Costs a one-time compile (~5-10 s for code_predictor, ~30-60 s for codec
  on first run).

The compile cost dominates **cold-start TTFC** under Config B (real codec).
Production warmup should run one synthetic forward at boot to amortise this.

---

## 7. The codec slowness story

**Symptom.** After wiring the real codec (commit `0f457c9`), first Gradio
"Generate" took ~10 minutes wall, and subsequent calls were also slow
(~5-15 s per utterance). Talker decode was fine; wall-time was inside
`codec.forward`.

**Diagnosis.** `qwen3_tts_components.py:_sliding_mask()` allocated a dense
`(T, T)` float `-inf` tensor every forward. For T=80 frames that's 6400
entries -- trivial alone. But: the codec pre_transformer has 8 layers each
calling `_sliding_mask` per forward, codec was being called per codec-frame
(80 frames x 8 layers = 640 calls per utterance), and SDPA with a float
additive mask does **not** take the fast path -- it does full dense matmul.
With a boolean mask SDPA short-circuits via Flash-style sparse attention.

**The fix.** `_sliding_mask` now returns a bool tensor with `True` outside
the band (~1 KB per call). The call site at line 596 already expected bool
semantics -- it was receiving the wrong type. After fix: codec forward
dropped from ~80 s per second utterance to ~0.5 s on a 2 s clip.

**Writing from scratch we'd:**

- Pre-allocate the mask once at codec init, slice per actual T. The mask is
  a pure function of T and the window; should not be recomputed per forward.
- Use FlexAttention (PyTorch 2.4+ block-sparse attention) -- expresses
  sliding-window directly without any mask materialisation.
- Batch codec calls at the source (we did this in `ui_v2`): collect N
  semantic tokens, run code_predictor and codec once on the batch. Trades a
  bit of TTFC for an order-of-magnitude throughput win.

---

## 8. What we'd build next

Prioritised by impact-per-effort:

1. **`torch.compile` warmup on service startup.** One synthetic forward
   through code_predictor + codec at process boot. Drops cold-start Config B
   TTFC from 694 ms to well under 100 ms. ~1 hour.

2. **Proper MRoPE in CUDA.** Replace the rotation loops in
   `csrc/kernel.cu:344-409` with per-axis position lookup. Closes the
   correctness gap in Section 4, unlocks real text-prefill in the kernel
   (Python prefill shortcut no longer needed). ~1-2 GPU hours; math specced
   in `e3-mrope-math.md`.

3. **Full e2e voice-loop validation.** `pipecat_demo.py` runs end-to-end on
   the Vast box, but we have not done a closed-loop "speak into mic, hear
   intelligible response" test. Blocked on (a) HF text-prefill numerical
   parity (Section 5) and (b) one CUDA-Graph capture quirk in the streaming
   yield path. ~half a day.

4. **Logits-diff correctness gate vs HF.** Emit pre-argmax logits via
   `LDG_DUMP_LOGITS` compile guard; assert `allclose(megakernel, hf,
   atol=1e-2)` on first 4 talker tokens. Cheaper than ear-test; would have
   caught Config B's intermittent silence early.

5. **Demo video.** 30-sec screen recording of the Pipecat voice loop. "It
   runs" reads weaker than a clip.

6. **Nsight Systems sweep.** At `hidden=2048` the prefetch knobs
   (`LDG_PREFETCH_*`) are tuned for 1024-wide tiles. Likely 5-15% headroom on
   the 1.988 ms/tok.

---

*Last updated: 2026-05-30. See `CHANGELOG.md` for sprint commit history,
`NOTICES.md` for license attribution, and the README for the perf table +
run recipe.*
