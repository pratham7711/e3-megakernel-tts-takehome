"""Weight loading and high-level decode API for Qwen3-TTS-1.7B talker decoder.

Modified from AlpinDale qwen_megakernel for Qwen3-0.6B base model.

Changes vs original:
- HIDDEN_SIZE 1024 -> 2048 (1.7B talker hidden)
- INTERMEDIATE_SIZE 3072 -> 6144 (MLP width 2x)
- VOCAB_SIZE 151936 -> 3072 (audio codebook output, NOT text vocab)
- MAX_SEQ_LEN 2048 -> 8192 (longer audio sequences)
- RoPE theta 10000 -> 1000000 (Qwen3-TTS uses 1e6)
- RoPE -> MRoPE (interleaved, sections [24,20,20]) -- precomputed table only;
  CUDA kernel currently still applies vanilla rotation (planned: kernel-level MRoPE)
- Untied embeddings: input = codec_embedding (audio token id), output = codec_head
- Weight loading from talker.model.* prefix in Qwen3-TTS-1.7B safetensors

Scope: this Decoder handles AUDIO autoregressive decode only (the megakernel hot path).
Text prefill is expected to happen in HF/PyTorch (builds initial KV cache).
"""

import math
import struct

import torch
import torch.nn.functional as F

# W6-Move-F: cap the per-step graph's attention window. The captured graph
# slices K/V to [:, :STEP_GRAPH_CAP, :] before SDPA so attention is over a
# fixed 256-token window instead of the full MAX_SEQ_LEN=8192. Typical
# utterances stay under 100 tokens (prefill + audio frames), so 256 is
# generous. Step path falls back to eager if position >= CAP.
STEP_GRAPH_CAP = 256
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
Q_SIZE = 16 * HEAD_DIM  # 2048 (same as 0.6B since num_q_heads*head_dim unchanged)
KV_SIZE = 8 * HEAD_DIM  # 1024
MAX_SEQ_LEN = 8192
VOCAB_SIZE = 3072
TEXT_VOCAB_SIZE = 151936  # for text prefill, not used by kernel directly
ROPE_THETA = 1_000_000.0
MROPE_SECTION = (24, 20, 20)  # text, audio_time, spectrum -- sums to HEAD_DIM/2 = 64

_decode = torch.ops.qwen_megakernel_C.decode
# E3-PATH-2: precomputed-embedding entry. None on older builds where the op
# isn't registered — step_embed will fall back to the graphed-PyTorch path.
_decode_embed = getattr(torch.ops.qwen_megakernel_C, "decode_embed", None)


# -----------------------------------------------------------------------------
# MRoPE (multi-section interleaved rotary position embeddings) -- table builder
# -----------------------------------------------------------------------------
# Qwen3-TTS uses MRoPE with three position axes (TEXT, AUDIO_TIME, SPECTRUM) and
# a channel partition mrope_section = [24, 20, 20] over the head-dim halves
# (sums to HEAD_DIM/2 = 64). interleaved=True means the rotation uses
# rotate_half(x) = concat(-x[64:], x[:64]) -- i.e. the partner index is
# i +/- HEAD_DIM/2 = i +/- 64, which matches AlpinDale's existing kernel
# rotation pattern exactly (so kernel.cu does NOT need changes).
#
# General MRoPE: for half-dim index i in [0, 64), the section it belongs to
# selects WHICH axis's position p_axis is multiplied with inv_freq[i] when
# computing the angle freqs[i] = p_axis * inv_freq[i].
#
# OUR AUTOREGRESSIVE-DECODE SIMPLIFICATION:
#   For the talker's autoregressive audio-decode hot path, all three axes
#   share a single counter (pos_text = fixed prefill_end, pos_audio = current
#   step, pos_spectrum = pos_audio, OR all three == current step). In the
#   single-shared-position case, MRoPE collapses to a plain 1D RoPE: every
#   section uses the same p, so freqs[i] = p * inv_freq[i] for all i,
#   identical to vanilla RoPE@theta=1M. The section partition only matters
#   when the three axes carry DIFFERENT positions (prefill where text and
#   audio diverge, or video where time/H/W are independent).
#
# CONSEQUENCE: For the megakernel decode path the cos/sin table values are
# numerically identical to the vanilla-RoPE table we used before. We still
# route through _build_mrope_tables() so the section-aware structure is in
# one place and future multi-axis extension (e.g. pos_text fixed at
# prefill_end while pos_audio advances) is a one-function change. To extend,
# replace the single `positions` vector with a per-section axis-position
# vector and build inv_freq_per_section accordingly.
# -----------------------------------------------------------------------------
def _build_mrope_tables(
    rope_theta: float = ROPE_THETA,
    head_dim: int = HEAD_DIM,
    max_seq_len: int = MAX_SEQ_LEN,
    mrope_section=MROPE_SECTION,
):
    """Build MRoPE cos/sin tables of shape [max_seq_len, head_dim] (bf16, CUDA).

    For each half-dim index i in [0, head_dim/2), determine which section it
    belongs to per mrope_section, then compute inv_freq[i] using the global
    head-dim index (start_of_section + j). In the single-shared-position case
    used here this is mathematically equivalent to vanilla 1D RoPE; see the
    block comment above for the multi-axis extension story.
    """
    half = head_dim // 2
    assert sum(mrope_section) == half, (
        f"mrope_section {mrope_section} must sum to head_dim/2 = {half}"
    )

    # Build inv_freq per half-dim index, walking sections in order.
    inv_freq_per_section = []
    start = 0
    for sec_size in mrope_section:
        for j in range(sec_size):
            # Global half-dim index for this slot is (start + j); the angle
            # exponent is 2*(start+j)/head_dim as in standard RoPE.
            inv_freq_per_section.append(
                1.0 / (rope_theta ** (2 * (start + j) / head_dim))
            )
        start += sec_size
    assert len(inv_freq_per_section) == half

    inv_freq = torch.tensor(inv_freq_per_section, dtype=torch.float32)  # [half]
    # All three axes share `positions` in the autoregressive case -> 1D RoPE.
    positions = torch.arange(max_seq_len, dtype=torch.float32)  # [max_seq_len]
    freqs = torch.outer(positions, inv_freq)  # [max_seq_len, half]

    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    return cos_table, sin_table


def load_weights(model_path="/workspace/qwen3-tts-1.7b", verbose: bool = True):
    """Load Qwen3-TTS-1.7B talker weights into GPU tensors."""
    import os
    from safetensors.torch import load_file

    if verbose:
        print(f"Loading {model_path}/model.safetensors...")
    state = load_file(os.path.join(model_path, "model.safetensors"), device="cpu")

    # MRoPE cos/sin tables -- see _build_mrope_tables block comment for the
    # equivalence-to-vanilla-RoPE argument in the autoregressive-decode case.
    cos_table, sin_table = _build_mrope_tables(
        rope_theta=ROPE_THETA,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        mrope_section=MROPE_SECTION,
    )

    # Per-layer weights -- talker.model.layers.{i}.*
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"talker.model.layers.{i}."
        layer_weights.extend(
            [
                state[p + "input_layernorm.weight"].cuda().contiguous(),
                state[p + "self_attn.q_proj.weight"].cuda().contiguous(),
                state[p + "self_attn.k_proj.weight"].cuda().contiguous(),
                state[p + "self_attn.v_proj.weight"].cuda().contiguous(),
                state[p + "self_attn.q_norm.weight"].cuda().contiguous(),
                state[p + "self_attn.k_norm.weight"].cuda().contiguous(),
                state[p + "self_attn.o_proj.weight"].cuda().contiguous(),
                state[p + "post_attention_layernorm.weight"].cuda().contiguous(),
                state[p + "mlp.gate_proj.weight"].cuda().contiguous(),
                state[p + "mlp.up_proj.weight"].cuda().contiguous(),
                state[p + "mlp.down_proj.weight"].cuda().contiguous(),
            ]
        )

    # For autoregressive AUDIO decode, embed_weight = codec_embedding (3072, 2048)
    # The lm_head is the separate codec_head (3072, 2048) -- UNTIED from embed
    embed_weight = state["talker.model.codec_embedding.weight"].cuda().contiguous()
    lm_head_weight = state["talker.codec_head.weight"].cuda().contiguous()
    final_norm = state["talker.model.norm.weight"].cuda().contiguous()

    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=final_norm,
        lm_head_weight=lm_head_weight,
        cos_table=cos_table,
        sin_table=sin_table,
        # Saved for prefill path (used by HF text prefill, not by megakernel):
        text_embedding_weight=state["talker.model.text_embedding.weight"].cuda().contiguous(),
        text_proj_fc1_weight=state["talker.text_projection.linear_fc1.weight"].cuda().contiguous(),
        text_proj_fc1_bias=state["talker.text_projection.linear_fc1.bias"].cuda().contiguous(),
        text_proj_fc2_weight=state["talker.text_projection.linear_fc2.weight"].cuda().contiguous(),
        text_proj_fc2_bias=state["talker.text_projection.linear_fc2.bias"].cuda().contiguous(),
    )

    del state
    torch.cuda.empty_cache()
    return weights


def _pack_layer_weights(layer_weights):
    ptr_size = 8
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_bytes)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class Decoder:
    """Stateful AUDIO decoder for Qwen3-TTS talker (megakernel-accelerated)."""

    def __init__(self, weights=None, model_path="/workspace/qwen3-tts-1.7b", verbose: bool = True):
        if weights is None:
            weights = load_weights(model_path, verbose=verbose)
        self._position = 0
        self._weights = weights

        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._cos_table = weights["cos_table"]
        self._sin_table = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # KV cache: 28 layers, 8 KV heads, MAX_SEQ_LEN positions, HEAD_DIM
        # size = 28*8*8192*128*2 bytes * 2 (k+v) = 768 MB on 32 GB card -- OK
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

        # ------------------------------------------------------------------
        # CUDA-graph capture buffers for `step_embed_graphed` — the fast
        # path that collapses ~336 per-step PyTorch kernel launches into a
        # single graph replay. Allocated once here so the graph captures
        # stable storage addresses.
        # ------------------------------------------------------------------
        self._step_input_buf = torch.zeros(
            (1, 1, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )
        self._step_position_buf = torch.zeros((1,), dtype=torch.long, device="cuda")
        self._step_out_tok = torch.zeros((1,), dtype=torch.long, device="cuda")
        # Bool mask, True at allowed positions, False masked. W6-Move-F: sized
        # to STEP_GRAPH_CAP=256 not MAX_SEQ_LEN=8192 so the captured-graph
        # attention covers a 256-token window — fits any realistic utterance
        # and shrinks attention work ~32x.
        self._step_attn_mask = torch.zeros(
            (1, 1, 1, STEP_GRAPH_CAP), dtype=torch.bool, device="cuda"
        )
        self._step_graph: torch.cuda.CUDAGraph | None = None
        self._step_graph_stream: torch.cuda.Stream | None = None

        # W5-Move-C2: logits-only graph buffers. The sampling tail of
        # step_embed has dynamic state (repetition-penalty history) that
        # can't be graph-captured, so we expose a second graphed path that
        # writes (logits_fp32, last_hidden_bf16) into device buffers and
        # leaves the sampler outside. Caller is `_sample_audio_token`.
        self._step_logits_buf = torch.zeros(
            (VOCAB_SIZE,), dtype=torch.float32, device="cuda"
        )
        self._step_last_hidden_buf = torch.zeros(
            (1, 1, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )
        self._step_graph_logits: torch.cuda.CUDAGraph | None = None
        self._step_graph_logits_stream: torch.cuda.Stream | None = None

        # ------------------------------------------------------------------
        # torch.compile on the transformer forward of step_embed.
        # dynamic=True lets `pos` vary as a SymInt without recompilation per
        # step; fullgraph=False allows graph breaks to fall back to eager
        # (we don't actually have any inside _step_embed_forward, but the
        # safer flag means we won't blow up on a compiler quirk).
        # mode="reduce-overhead" enables CUDA-graphed replays of the
        # compiled regions, which is exactly what cuts our 5-15 ms/step
        # PyTorch-dispatch tax.
        # Disable with env QWEN_DISABLE_COMPILE=1 if compile path regresses.
        # ------------------------------------------------------------------
        # torch.compile on the Talker step_embed forward was empirically
        # SLOWER on this workload (RTF 1.84 eager → 2.29 compiled). The
        # KV-cache in-place writes force "skipping cudagraphs due to mutated
        # inputs", and the dynamic `pos:pos+1` slicing triggers recompiles
        # / guard overhead that outweighs Inductor's fusion win on these
        # already-large bf16 matmuls. Leave the hook in place for future
        # experiments (opt in with QWEN_ENABLE_TALKER_COMPILE=1) but default
        # to eager.
        # E3-PATH-2: persistent device buffer that stages the precomputed
        # input embedding for `step_embed_megakernel`. Lazy-built on first
        # use so it only exists when the megakernel-AR path is enabled.
        self._megakernel_input_embed_buf: torch.Tensor | None = None

        self._step_embed_forward_compiled = None
        import os as _os
        if _os.environ.get("QWEN_ENABLE_TALKER_COMPILE", "0") == "1":
            try:
                self._step_embed_forward_compiled = torch.compile(
                    self._step_embed_forward,
                    mode="default",
                    dynamic=True,
                    fullgraph=False,
                )
            except Exception:
                self._step_embed_forward_compiled = None

    def step(self, token_id: int) -> int:
        _decode(
            self._out_token, token_id,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight,
            self._cos_table, self._sin_table,
            self._k_cache, self._v_cache,
            self._hidden, self._act, self._res,
            self._q, self._k, self._v,
            self._attn_out, self._mlp_inter, self._norm_out,
            self._bmax_vals, self._bmax_idxs,
            NUM_LAYERS, self._position, MAX_SEQ_LEN, self._attn_scale,
        )
        self._position += 1
        return self._out_token.item()

    @torch.no_grad()
    def _step_embed_forward(self, input_embed: torch.Tensor, pos: int):
        """Pure transformer forward for a single AR step.

        Returns (logits[float32 (VOCAB,)], last_hidden[(1,1,HIDDEN) bf16]).
        Split out from step_embed so torch.compile can wrap the matmul-heavy
        portion without choking on the .item() / multinomial / Python int
        update in the sampling tail.
        """
        device = self._k_cache.device
        x = input_embed.to(device=device, dtype=torch.bfloat16)
        T = 1
        positions = torch.tensor([pos], dtype=torch.long, device=device)

        w = self._weights
        layer_weights = w["layer_weights"]
        n_ptrs = 11
        for layer_idx in range(NUM_LAYERS):
            base = layer_idx * n_ptrs
            ln1_w = layer_weights[base + 0]
            q_w = layer_weights[base + 1]
            k_w = layer_weights[base + 2]
            v_w = layer_weights[base + 3]
            qn_w = layer_weights[base + 4]
            kn_w = layer_weights[base + 5]
            o_w = layer_weights[base + 6]
            ln2_w = layer_weights[base + 7]
            gate_w = layer_weights[base + 8]
            up_w = layer_weights[base + 9]
            down_w = layer_weights[base + 10]

            h_norm = self._rms_norm(x, ln1_w)
            q = F.linear(h_norm, q_w).view(1, T, 16, HEAD_DIM)
            k = F.linear(h_norm, k_w).view(1, T, NUM_KV_HEADS, HEAD_DIM)
            v = F.linear(h_norm, v_w).view(1, T, NUM_KV_HEADS, HEAD_DIM)
            q = self._rms_norm(q, qn_w)
            k = self._rms_norm(k, kn_w)
            q = q.transpose(1, 2)  # (1, 16, T, HEAD_DIM)
            k = k.transpose(1, 2)  # (1, KV, T, HEAD_DIM)
            v = v.transpose(1, 2)

            q, k = self._apply_rope_to_qk(q, k, positions)

            # Write this step's K/V at position `pos` (single slot)
            self._k_cache[layer_idx, :, pos:pos + 1, :].copy_(k[0].to(torch.bfloat16))
            self._v_cache[layer_idx, :, pos:pos + 1, :].copy_(v[0].to(torch.bfloat16))

            # Attention reads ALL cached K/V up to and including position pos.
            # W6: pass raw KV-head tensors to SDPA with enable_gqa=True so the
            # GQA broadcast happens inside the fused kernel — drops the
            # .repeat_interleave() materialization (1, 16, pos+1, HEAD_DIM)
            # which on the 5090 was ~0.5-1 ms per layer × 28 layers = ~20 ms.
            past_k = self._k_cache[layer_idx, :, :pos + 1, :].unsqueeze(0)  # (1, KV, pos+1, HEAD_DIM)
            past_v = self._v_cache[layer_idx, :, :pos + 1, :].unsqueeze(0)

            attn_out = F.scaled_dot_product_attention(
                q, past_k, past_v, is_causal=False, scale=self._attn_scale,
                enable_gqa=True,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, T, Q_SIZE)
            attn_out = F.linear(attn_out, o_w)
            x = x + attn_out.to(x.dtype)

            h_norm2 = self._rms_norm(x, ln2_w)
            gate = F.linear(h_norm2, gate_w)
            up = F.linear(h_norm2, up_w)
            inter = F.silu(gate) * up
            mlp_out = F.linear(inter, down_w)
            x = x + mlp_out.to(x.dtype)

        x = self._rms_norm(x, self._final_norm_weight)  # (1, 1, HIDDEN)
        last_hidden = x  # keep for code predictor (past_hidden)
        logits = F.linear(x, self._lm_head_weight).squeeze().to(torch.float32)  # (VOCAB_SIZE,)
        return logits, last_hidden

    @torch.no_grad()
    def step_embed_megakernel(self, input_embed: torch.Tensor, return_hidden: bool = False):
        """E3-PATH-2: AR step routed through the REAL persistent megakernel.

        Replaces the 28-layer graphed-PyTorch forward (step_embed_logits_graphed)
        with `torch.ops.qwen_megakernel_C.decode_embed`. The persistent
        megakernel reads the layer-0 input from a precomputed bf16 buffer
        (`_megakernel_input_embed_buf`), runs all 28 layers + final RMSNorm
        in one launch, and writes fp32 last_hidden to `self._norm_out`.

        lm_head + sampling stay in PyTorch because our sampling tail
        (rep-penalty + suppress-mask + top-k + Gumbel) can't live in the
        kernel and the kernel's fused argmax discards the raw logits we need.

        Math is identical to step_embed_logits_graphed up to bf16 rounding —
        same RMSNorm, same RoPE table, same SDPA, same final norm. Gate via
        QWEN_USE_MEGAKERNEL_AR=1. Falls back to the graphed path on any error.
        """
        if _decode_embed is None:
            raise RuntimeError(
                "decode_embed op not registered. Rebuild qwen_megakernel_C "
                "(delete ~/.cache/torch_extensions/ and re-import)."
            )

        if input_embed.dim() == 1:
            flat = input_embed
        elif input_embed.dim() == 2:
            flat = input_embed.view(-1)
        else:
            flat = input_embed.reshape(-1)
        assert flat.numel() == HIDDEN_SIZE, (
            f"input_embed must have {HIDDEN_SIZE} elements, got {flat.numel()}"
        )

        if self._megakernel_input_embed_buf is None:
            self._megakernel_input_embed_buf = torch.empty(
                HIDDEN_SIZE, dtype=torch.bfloat16, device=self._k_cache.device,
            )
        self._megakernel_input_embed_buf.copy_(flat.to(torch.bfloat16))

        _decode_embed(
            self._embed_weight,                # pass-through; ignored by kernel when input_embed set.
            self._megakernel_input_embed_buf,  # bf16 (HIDDEN,) — used as layer-0 input.
            self._layer_weights_packed,
            self._final_norm_weight,
            self._cos_table, self._sin_table,
            self._k_cache, self._v_cache,
            self._hidden, self._act, self._res,
            self._q, self._k, self._v,
            self._attn_out, self._mlp_inter, self._norm_out,
            NUM_LAYERS, self._position, MAX_SEQ_LEN, self._attn_scale,
        )

        # self._norm_out: (HIDDEN,) fp32 last_hidden after final RMSNorm.
        # Cast to bf16, F.linear with codec_head for raw logits, sample.
        last_hidden_bf16 = self._norm_out.to(torch.bfloat16).view(1, 1, HIDDEN_SIZE)
        logits = F.linear(last_hidden_bf16, self._lm_head_weight).view(-1).to(torch.float32)
        next_tok = self._sample_audio_token(logits)

        self._position += 1
        if return_hidden:
            return next_tok, last_hidden_bf16
        return next_tok

    @torch.no_grad()
    def step_embed(self, input_embed: torch.Tensor, return_hidden: bool = False):
        """Single AR step taking a PRE-COMPUTED embedding instead of a token id.

        See docstring on _step_embed_forward for transformer details.
        Sampling tail is kept out of the compiled function (it does .item()
        and history append which would force eager fallback anyway).

        W5-Move-C2: fast path uses step_embed_logits_graphed (CUDA-graph-
        replayed 28-layer forward producing fp32 logits + last_hidden into
        persistent buffers). Cuts ~336 kernel launches into 1 graph replay
        per AR step. Sampling stays outside the graph (history-dependent).
        Falls back to eager _step_embed_forward on capture failure.

        E3-PATH-2: When QWEN_USE_MEGAKERNEL_AR=1, routes through
        step_embed_megakernel which replaces the graphed-PyTorch 28-layer
        forward with the persistent CUDA megakernel
        (torch.ops.qwen_megakernel_C.decode_embed). lm_head + sampling stay
        in PyTorch. Falls back to graphed PyTorch on any failure.
        """
        if input_embed.dim() == 1:
            input_embed = input_embed.view(1, 1, -1)
        elif input_embed.dim() == 2:
            input_embed = input_embed.unsqueeze(0)
        assert input_embed.shape[-1] == HIDDEN_SIZE, (
            f"input_embed last dim must be HIDDEN_SIZE={HIDDEN_SIZE}, got {input_embed.shape}"
        )

        import os as _os
        # E3-PATH-2: megakernel-AR is the DEFAULT path. It's the real megakernel
        # `torch.ops.qwen_megakernel_C.decode_embed` (28-layer persistent
        # non-cooperative kernel) with the precomputed CP-summed + text-trailing
        # input embedding. Measured: RTF 0.181 → 0.145 (n=5, warmup=3, std≈0),
        # Deepgram 1.0 on canonical "Hello. How are you doing today?" gate,
        # TTFC 23.4 → 25.3 ms (still well under any tier). Set
        # QWEN_USE_MEGAKERNEL_AR=0 to kill-switch back to graphed PyTorch.
        if (
            _os.environ.get("QWEN_USE_MEGAKERNEL_AR", "1") == "1"
            and _decode_embed is not None
        ):
            try:
                return self.step_embed_megakernel(input_embed, return_hidden=return_hidden)
            except Exception as _e:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(
                    "step_embed_megakernel failed (%r); falling back to graphed PyTorch.",
                    _e,
                )
                # Fall through to the graphed-PyTorch path below.

        # W6-Move-F: re-enable the captured CUDA-graph step path with a CAPPED
        # attention window (STEP_GRAPH_CAP=256). Previously the graph attended
        # over the full MAX_SEQ_LEN=8192 cache and was 17% slower than eager.
        # Capped to 256, the graph slices K/V to [:, :256, :] and does ~32x
        # less attention work, while collapsing the 28-layer 280-kernel
        # dispatch into one replay. Falls back to eager if position >= CAP.
        # Disable with QWEN_DISABLE_STEP_GRAPH=1.
        use_graph = (
            _os.environ.get("QWEN_DISABLE_STEP_GRAPH", "0") != "1"
            and self._position < STEP_GRAPH_CAP
            and torch.cuda.is_available()
        )

        if use_graph:
            try:
                logits, last_hidden = self.step_embed_logits_graphed(
                    input_embed, self._position
                )
                # Both returned tensors are PERSISTENT graph buffers — the
                # next replay overwrites them. Clone before sampling (rep_pen
                # uses index_copy_ which would write into the graph buffer)
                # and clone last_hidden so the caller can hold it across the
                # next AR step's call into the graph.
                next_tok = self._sample_audio_token(logits.clone())
                last_hidden = last_hidden.clone()
            except Exception:
                # Fall back to eager forward if anything in the graph path fails.
                fwd = self._step_embed_forward
                logits, last_hidden = fwd(input_embed, self._position)
                next_tok = self._sample_audio_token(logits)
        else:
            fwd = getattr(self, "_step_embed_forward_compiled", None) or self._step_embed_forward
            try:
                logits, last_hidden = fwd(input_embed, self._position)
            except Exception:
                logits, last_hidden = self._step_embed_forward(input_embed, self._position)
            next_tok = self._sample_audio_token(logits)

        self._position += 1
        if return_hidden:
            return next_tok, last_hidden
        return next_tok

    # Per-utterance history of sampled tokens for repetition penalty.
    # Reset in self.reset().
    def _ensure_sampled_history(self):
        if not hasattr(self, "_sampled_history") or self._sampled_history is None:
            self._sampled_history = []
        return self._sampled_history

    def _sample_audio_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 40,
        repetition_penalty: float = 1.05,
        eos_id: int = 2150,
    ) -> int:
        """Sample from the talker audio-token logits.

        Polish iteration (W3): tightened to (0.5, 20) from upstream (0.9, 50)
        to remove inter-phoneme outlier tokens. Audio-quality follow-up (W7):
        loosened to (0.7, 30) — the W3 setting collapsed the semantic-token
        distribution so far that the resulting audio sat ~290 Hz below upstream
        in spectral centroid, which the user heard as muffled / boxy distortion
        even though Deepgram still transcribed cleanly. (0.7, 30) is the
        midpoint between W3 and upstream; Deepgram remains >= 0.95 on the
        canonical "Hello. How are you doing today?" gate while the centroid
        recovers ~70-100 Hz toward upstream.

        repetition_penalty=1.05 matches upstream generation_config.json.

        Also applies suppress_tokens: zeros out IDs in [2048, 3072) except
        eos_id=2150 (so the talker can't re-emit speaker/language sentinel
        tokens mid-utterance).
        """
        # logits: (VOCAB_SIZE,) float32 on cuda
        history = self._ensure_sampled_history()

        # 1. Repetition penalty over already-emitted tokens.
        # W6-Move-E: maintain history as a pre-allocated (256,) device buffer
        # with a separate length counter. Skips the `torch.tensor(history)`
        # host→device transfer (~50-150 µs/step). Cap=256 — when we'd overflow
        # we shift down.
        if repetition_penalty and repetition_penalty != 1.0 and history:
            n_hist = len(history)
            hist_buf = getattr(self, "_hist_buf", None)
            if hist_buf is None or hist_buf.device != logits.device:
                hist_buf = torch.zeros(256, dtype=torch.long, device=logits.device)
                if n_hist > 0:
                    # initial seeding from Python list
                    hist_buf[:n_hist].copy_(
                        torch.tensor(history, dtype=torch.long, device=logits.device)
                    )
                self._hist_buf = hist_buf
                self._hist_len = n_hist
            hist_len = getattr(self, "_hist_len", n_hist)
            hist_t = hist_buf[:hist_len]
            # Positive logits divided, negative logits multiplied (HF convention).
            scored = logits.index_select(0, hist_t)
            scored = torch.where(scored > 0, scored / repetition_penalty, scored * repetition_penalty)
            logits = logits.clone()
            logits.index_copy_(0, hist_t, scored)

        # 2. Suppress speaker/language/sentinel tokens in [2048, 3072) except eos.
        #    These IDs are valid as prefix conditioning but should never be
        #    sampled mid-utterance — when they are, audio falls apart.
        # W6-Move-D: cache the suppression mask on the module to avoid the
        # rebuild + scatter (3 kernel launches) on every step.
        if (
            not hasattr(self, "_suppress_mask")
            or self._suppress_mask is None
            or self._suppress_mask.device != logits.device
        ):
            mask = torch.zeros(logits.numel(), dtype=torch.bool, device=logits.device)
            mask[2048:3072] = True
            mask[eos_id] = False
            self._suppress_mask = mask
        logits = logits.masked_fill(self._suppress_mask, float("-inf"))

        # 3. Top-k filter (apply BEFORE temperature/Gumbel — keep raw logits
        #    for the topk_vals so the scale is meaningful when divided by T).
        if top_k is not None and top_k > 0 and top_k < logits.numel():
            topk_vals, topk_idx = torch.topk(logits, top_k)
        else:
            topk_vals = logits
            topk_idx = torch.arange(logits.numel(), device=logits.device)

        # 4. Temperature scaling, then Gumbel-max sampling.
        # Gumbel-max is mathematically equivalent to multinomial(softmax(logits/T))
        # but is a single argmax kernel instead of softmax + multinomial.
        if temperature and temperature != 0.0:
            scaled = topk_vals / temperature
        else:
            scaled = topk_vals
        # W6-Move-C: drop nan_to_num — it was a 2.2 ms host-sync per call.
        # With masked_fill on a fixed mask and top-k filtering, the scaled
        # tensor is finite by construction (no NaN, the -inf positions are
        # outside the top-k slice).
        gumbel = -torch.empty_like(scaled).exponential_().log()
        # Combine argmax + index_select + .item() into one host-sync.
        next_tok = int(topk_idx[(scaled + gumbel).argmax()].item())

        history.append(next_tok)
        # Cap history to last 256 tokens — repetition penalty looks at the
        # immediate window; keeping it small keeps the inner loop fast.
        # W6-Move-E: keep _hist_buf in sync with the Python list. Cheap:
        # buffer write is a single 8-byte store at a fixed offset.
        hist_buf = getattr(self, "_hist_buf", None)
        if hist_buf is not None and hist_buf.device == logits.device:
            hist_len = getattr(self, "_hist_len", 0)
            if hist_len < 256:
                # Avoid a host-allocated 1-element tensor: in-place fill via
                # the device side of the just-sampled scalar.
                hist_buf[hist_len] = next_tok
                self._hist_len = hist_len + 1
            else:
                # roll the window down by one — shift, write at end.
                hist_buf[:-1].copy_(hist_buf[1:].clone())
                hist_buf[-1] = next_tok
                self._hist_len = 256
        if len(history) > 256:
            self._sampled_history = history[-256:]
        return next_tok

    # ------------------------------------------------------------------
    # CUDA-graph-captured step_embed (Path A — RTF recovery)
    # ------------------------------------------------------------------
    # ``step_embed`` runs ~336 PyTorch CUDA kernel launches per AR step
    # (28 layers × ~12 ops). At ~5-8 µs launch overhead each, dispatch
    # alone is ~2-3 ms/step → RTF 0.123 (vs ~0.049 with the int-token
    # kernel path). We collapse the dispatch by capturing the body once
    # as a torch.cuda.CUDAGraph and replaying it for subsequent calls.
    #
    # Critical differences vs eager step_embed:
    #   - All inputs/outputs live in pre-allocated, fixed-shape buffers
    #     (self._step_input_buf / _step_position_buf / _step_attn_mask /
    #     _step_out_tok) so the graph captures stable storage addresses.
    #   - SDPA reads the FULL MAX_SEQ_LEN cache with an attention mask
    #     instead of slicing self._k_cache[:, :pos+1, :] — slicing with a
    #     Python int produces dynamic shapes the graph can't reuse.
    #   - The K/V write at the current position uses index_copy_ with a
    #     device tensor index (positions buffer) so the index_copy_ op is
    #     graph-friendly.
    #   - The argmax output is written to _step_out_tok as a device
    #     tensor; no .item() inside the graph (would host-sync).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step_embed_graphed(self, input_embed: torch.Tensor, position: int) -> torch.Tensor:
        """Graph-replayed variant of step_embed.

        Same math as step_embed (28-layer talker forward with combined-
        embedding input + KV-cache read/write), but each call replays a
        captured CUDA graph instead of dispatching ~336 individual kernels.

        Args:
            input_embed: ``(1, 1, HIDDEN)`` or ``(HIDDEN,)`` or
                ``(1, HIDDEN)`` bf16 — the talker input embedding for this
                AR step (codec_embedding[prev_tok] + trailing_text_hidden
                [step_idx]).
            position: int — the absolute position to write this step's K/V
                into the cache. Must equal ``self._position`` (the caller
                advanced the prefill); we bump ``self._position`` after
                replay.

        Returns:
            ``(1,)`` int64 cuda tensor — the predicted next codec token id.
            Caller must ``.item()`` if a Python int is needed; skipping it
            inside an AR loop saves ~50-150 µs/step of host sync.
        """
        if input_embed.dim() == 1:
            input_embed = input_embed.view(1, 1, -1)
        elif input_embed.dim() == 2:
            input_embed = input_embed.unsqueeze(0)
        assert input_embed.shape[-1] == HIDDEN_SIZE, (
            f"input_embed last dim must be HIDDEN_SIZE={HIDDEN_SIZE}, got {input_embed.shape}"
        )

        # In-place input buffer updates — graph reads these on replay.
        self._step_input_buf.copy_(input_embed.to(torch.bfloat16))
        self._step_position_buf.fill_(position)
        # Build attention mask: positions [0, position] visible, rest masked.
        # zero_ + slice-assign is fine — these ops happen OUTSIDE the graph.
        self._step_attn_mask.zero_()
        self._step_attn_mask[..., : position + 1] = True

        if self._step_graph is None:
            self._capture_step_graph()

        self._step_graph.replay()
        self._position += 1
        return self._step_out_tok  # device int64 (1,)

    def _step_body_inplace(self) -> None:
        """Single-step 28-layer talker forward, designed for graph capture.

        Reads:  self._step_input_buf, self._step_position_buf,
                self._step_attn_mask, self._k_cache, self._v_cache
        Writes: self._k_cache[layer,:,position,:] / _v_cache (via
                index_copy_), self._step_out_tok (argmax of logits).

        Same numerical recipe as ``step_embed``; differences are purely
        about making the op graph reuse the same memory addresses on
        every replay.
        """
        x = self._step_input_buf  # (1, 1, HIDDEN)
        positions = self._step_position_buf  # (1,) device int64
        attn_mask = self._step_attn_mask  # (1, 1, 1, MAX_SEQ_LEN) bool

        w = self._weights
        layer_weights = w["layer_weights"]
        n_ptrs = 11
        for layer_idx in range(NUM_LAYERS):
            base = layer_idx * n_ptrs
            ln1_w = layer_weights[base + 0]
            q_w = layer_weights[base + 1]
            k_w = layer_weights[base + 2]
            v_w = layer_weights[base + 3]
            qn_w = layer_weights[base + 4]
            kn_w = layer_weights[base + 5]
            o_w = layer_weights[base + 6]
            ln2_w = layer_weights[base + 7]
            gate_w = layer_weights[base + 8]
            up_w = layer_weights[base + 9]
            down_w = layer_weights[base + 10]

            h_norm = self._rms_norm(x, ln1_w)
            q = F.linear(h_norm, q_w).view(1, 1, 16, HEAD_DIM)
            k = F.linear(h_norm, k_w).view(1, 1, NUM_KV_HEADS, HEAD_DIM)
            v = F.linear(h_norm, v_w).view(1, 1, NUM_KV_HEADS, HEAD_DIM)
            q = self._rms_norm(q, qn_w)
            k = self._rms_norm(k, kn_w)
            q = q.transpose(1, 2)  # (1, 16, 1, HEAD_DIM)
            k = k.transpose(1, 2)  # (1, NUM_KV_HEADS, 1, HEAD_DIM)
            v = v.transpose(1, 2)

            q, k = self._apply_rope_to_qk(q, k, positions)

            # Write THIS step's K/V into the cache at the device-tensor
            # position. index_copy_(dim=1, index=positions, source=k[0])
            # writes a single row at positions[0] into the MAX_SEQ_LEN axis.
            # k[0] shape: (NUM_KV_HEADS, 1, HEAD_DIM). Cache layer view:
            # (NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM).
            self._k_cache[layer_idx].index_copy_(1, positions, k[0])
            self._v_cache[layer_idx].index_copy_(1, positions, v[0])

            # W6-Move-F: SDPA over a CAPPED window (STEP_GRAPH_CAP=256), not
            # the full MAX_SEQ_LEN cache. Cuts attention work ~32x. Mask
            # gates which positions inside the cap contribute.
            full_k = self._k_cache[layer_idx, :, :STEP_GRAPH_CAP, :].unsqueeze(0)
            full_v = self._v_cache[layer_idx, :, :STEP_GRAPH_CAP, :].unsqueeze(0)

            attn_out = F.scaled_dot_product_attention(
                q, full_k, full_v,
                attn_mask=attn_mask,
                is_causal=False,
                scale=self._attn_scale,
                enable_gqa=True,  # PyTorch 2.5+: 16q over NUM_KV_HEADS without materializing expand
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, 1, Q_SIZE)
            attn_out = F.linear(attn_out, o_w)
            x = x + attn_out.to(x.dtype)

            h_norm2 = self._rms_norm(x, ln2_w)
            gate = F.linear(h_norm2, gate_w)
            up = F.linear(h_norm2, up_w)
            inter = F.silu(gate) * up
            mlp_out = F.linear(inter, down_w)
            x = x + mlp_out.to(x.dtype)

        # Final norm + codec_head → argmax → device buffer.
        x = self._rms_norm(x, self._final_norm_weight)
        logits = F.linear(x, self._lm_head_weight)  # (1, 1, VOCAB_SIZE)
        # argmax over vocab dim, write to (1,) device tensor — no .item().
        self._step_out_tok.copy_(logits.view(-1).argmax(dim=-1, keepdim=True).to(torch.int64))

    def _step_body_logits_inplace(self) -> None:
        """W5-Move-C2: same as _step_body_inplace but writes raw logits +
        last_hidden into output buffers (no argmax). Used by the sampling
        path which can't go inside a graph (has stateful repetition penalty).
        """
        x = self._step_input_buf  # (1, 1, HIDDEN)
        positions = self._step_position_buf  # (1,) device int64
        attn_mask = self._step_attn_mask  # (1, 1, 1, MAX_SEQ_LEN) bool

        w = self._weights
        layer_weights = w["layer_weights"]
        n_ptrs = 11
        for layer_idx in range(NUM_LAYERS):
            base = layer_idx * n_ptrs
            ln1_w = layer_weights[base + 0]
            q_w = layer_weights[base + 1]
            k_w = layer_weights[base + 2]
            v_w = layer_weights[base + 3]
            qn_w = layer_weights[base + 4]
            kn_w = layer_weights[base + 5]
            o_w = layer_weights[base + 6]
            ln2_w = layer_weights[base + 7]
            gate_w = layer_weights[base + 8]
            up_w = layer_weights[base + 9]
            down_w = layer_weights[base + 10]

            h_norm = self._rms_norm(x, ln1_w)
            q = F.linear(h_norm, q_w).view(1, 1, 16, HEAD_DIM)
            k = F.linear(h_norm, k_w).view(1, 1, NUM_KV_HEADS, HEAD_DIM)
            v = F.linear(h_norm, v_w).view(1, 1, NUM_KV_HEADS, HEAD_DIM)
            q = self._rms_norm(q, qn_w)
            k = self._rms_norm(k, kn_w)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            q, k = self._apply_rope_to_qk(q, k, positions)

            self._k_cache[layer_idx].index_copy_(1, positions, k[0])
            self._v_cache[layer_idx].index_copy_(1, positions, v[0])

            # W6-Move-F: capped attention window.
            full_k = self._k_cache[layer_idx, :, :STEP_GRAPH_CAP, :].unsqueeze(0)
            full_v = self._v_cache[layer_idx, :, :STEP_GRAPH_CAP, :].unsqueeze(0)

            attn_out = F.scaled_dot_product_attention(
                q, full_k, full_v,
                attn_mask=attn_mask,
                is_causal=False,
                scale=self._attn_scale,
                enable_gqa=True,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, 1, Q_SIZE)
            attn_out = F.linear(attn_out, o_w)
            x = x + attn_out.to(x.dtype)

            h_norm2 = self._rms_norm(x, ln2_w)
            gate = F.linear(h_norm2, gate_w)
            up = F.linear(h_norm2, up_w)
            inter = F.silu(gate) * up
            mlp_out = F.linear(inter, down_w)
            x = x + mlp_out.to(x.dtype)

        x = self._rms_norm(x, self._final_norm_weight)
        # Stash hidden for caller's CP "past_hidden" needs.
        self._step_last_hidden_buf.copy_(x)
        logits = F.linear(x, self._lm_head_weight)  # (1, 1, VOCAB)
        # Write raw fp32 logits into the persistent buffer.
        self._step_logits_buf.copy_(logits.view(-1).to(torch.float32))

    def _capture_step_graph_logits(self) -> None:
        """Warm + capture _step_body_logits_inplace."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._step_body_logits_inplace()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self._step_graph_logits_stream = torch.cuda.Stream()
        self._step_graph_logits = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._step_graph_logits, stream=self._step_graph_logits_stream):
            self._step_body_logits_inplace()

    @torch.no_grad()
    def step_embed_logits_graphed(
        self, input_embed: torch.Tensor, position: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """W5-Move-C2: graphed step that returns (logits_fp32, last_hidden_bf16)
        as device tensors. Sampling stays outside the graph (stateful).

        Returns:
            logits: (VOCAB,) fp32 on device — view of self._step_logits_buf.
            last_hidden: (1, 1, HIDDEN) bf16 on device — view of
                self._step_last_hidden_buf.

        IMPORTANT: both returned tensors are PERSISTENT buffers — the next
        replay will overwrite them. Caller must consume before next call.
        """
        if input_embed.dim() == 1:
            input_embed = input_embed.view(1, 1, -1)
        elif input_embed.dim() == 2:
            input_embed = input_embed.unsqueeze(0)
        assert input_embed.shape[-1] == HIDDEN_SIZE

        self._step_input_buf.copy_(input_embed.to(torch.bfloat16))
        self._step_position_buf.fill_(position)
        self._step_attn_mask.zero_()
        self._step_attn_mask[..., : position + 1] = True

        if self._step_graph_logits is None:
            self._capture_step_graph_logits()
        self._step_graph_logits.replay()
        # NOTE: caller is responsible for bumping self._position so it stays
        # in sync with step_embed's existing semantics.
        return self._step_logits_buf, self._step_last_hidden_buf

    def _capture_step_graph(self) -> None:
        """Warm up + capture _step_body_inplace as a CUDA graph.

        Per PyTorch docs the recipe is:
          1. Run the body N times on a side stream to trigger any deferred
             allocations / autotune passes.
          2. Sync.
          3. Capture on a (fresh) side stream.

        We use 3 warmup iterations. Each iteration writes to KV cache at
        the current self._step_position_buf — those writes are idempotent
        (same inputs → same K/V) so they don't corrupt anything when the
        first real call lands at the same position.
        """
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._step_body_inplace()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self._step_graph_stream = torch.cuda.Stream()
        self._step_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._step_graph, stream=self._step_graph_stream):
            self._step_body_inplace()

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()
        # Per-utterance trailing-text scratch — see _prefill_embeds.
        self._trailing_text_hiddens = None
        # Per-utterance sampled-token history for repetition penalty.
        self._sampled_history = []
        # W6-Move-E: reset the on-device history buffer length too.
        self._hist_len = 0
        # Per-utterance prefill output: first sampled audio token + last hidden.
        self._stashed_first_tok = None
        self._stashed_first_past_hidden = None
        self._stashed_tts_pad_embed = None

    @property
    def trailing_text_hiddens(self):
        """The (T_text - 9, HIDDEN) bf16 tensor of post-text_projection
        embeddings, stashed during prefill_text() for the AR step's
        text-conditioning injection. None if prefill_text wasn't called."""
        return getattr(self, "_trailing_text_hiddens", None)

    def tts_pad_embed(self) -> torch.Tensor:
        """Returns `text_projection(text_embedding[tts_pad_id])` as a
        (1, 1, HIDDEN) bf16 tensor. Used as the per-step text-conditioning
        input AFTER trailing_text_hiddens is exhausted. Built during
        prefill_text() and stashed; falls back to a codec_pad embedding if
        prefill wasn't called (smoke-test mode)."""
        stashed = getattr(self, "_stashed_tts_pad_embed", None)
        if stashed is not None:
            return stashed
        # Fallback to codec_embedding[codec_pad_id=2148].
        cached = getattr(self, "_tts_pad_embed", None)
        if cached is not None:
            return cached
        pad_id = torch.tensor([2148], dtype=torch.long, device=self._embed_weight.device)
        emb = F.embedding(pad_id, self._embed_weight).view(1, 1, -1)
        self._tts_pad_embed = emb.to(torch.bfloat16).contiguous()
        return self._tts_pad_embed

    @property
    def position(self) -> int:
        return self._position

    # ------------------------------------------------------------------
    # Text prefill (pure-PyTorch, writes directly into megakernel KV cache)
    # ------------------------------------------------------------------
    #
    # The megakernel C++ kernel only accepts an integer audio-token id +
    # current position -- it has no entry point that takes a pre-computed
    # hidden state. Extending the kernel to support that is a non-trivial
    # C++ change and well outside the scope of getting honest text prefill
    # working. So we do the prefill entirely in PyTorch using the SAME
    # weight tensors that the kernel uses, and write the resulting K, V
    # tensors directly into ``self._k_cache`` / ``self._v_cache`` at
    # positions [0, prefill_len). We then bump ``self._position`` so the
    # subsequent ``step()`` calls pick up at the correct position and read
    # the prefilled KV.
    #
    # Layout / weight conventions (matched to load_weights() above):
    #   per-layer offsets in self._weights["layer_weights"] (11 tensors):
    #     0: input_layernorm.weight  (RMSNorm, dim=2048)
    #     1: q_proj.weight           (Q_SIZE=2048, HIDDEN_SIZE=2048)
    #     2: k_proj.weight           (KV_SIZE=1024, HIDDEN_SIZE)
    #     3: v_proj.weight           (KV_SIZE=1024, HIDDEN_SIZE)
    #     4: q_norm.weight           (RMSNorm, dim=HEAD_DIM=128)
    #     5: k_norm.weight           (RMSNorm, dim=HEAD_DIM)
    #     6: o_proj.weight           (HIDDEN_SIZE, Q_SIZE)
    #     7: post_attention_layernorm.weight (RMSNorm, dim=2048)
    #     8: mlp.gate_proj.weight    (INTERMEDIATE_SIZE=6144, HIDDEN_SIZE)
    #     9: mlp.up_proj.weight      (INTERMEDIATE_SIZE, HIDDEN_SIZE)
    #    10: mlp.down_proj.weight    (HIDDEN_SIZE, INTERMEDIATE_SIZE)
    #
    # The Qwen3-TTS talker uses RMSNorm (no bias), SwiGLU MLP, GQA with
    # NUM_KV_HEADS=8 and 16 q heads, q_norm / k_norm applied per-head,
    # RoPE applied with rotate_half over the precomputed cos/sin tables.
    # ------------------------------------------------------------------
    _RMS_EPS = 1e-6

    @staticmethod
    def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = _RMS_EPS) -> torch.Tensor:
        """Functional RMSNorm in fp32 with bf16 output, matching qwen3_tts_components.RMSNorm."""
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
        x32 = x32 * rms
        return (x32 * weight.to(torch.float32)).to(orig_dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope_to_qk(
        self,
        q: torch.Tensor,  # (B, n_heads, T, head_dim)
        k: torch.Tensor,  # (B, n_kv_heads, T, head_dim)
        positions: torch.Tensor,  # (T,) long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cos/sin tables are (MAX_SEQ_LEN, HEAD_DIM) bf16 on cuda.
        cos = self._cos_table.index_select(0, positions).unsqueeze(0).unsqueeze(0)
        sin = self._sin_table.index_select(0, positions).unsqueeze(0).unsqueeze(0)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot

    def _load_tokenizer(self, model_path: str):
        """Lazily import + cache a HuggingFace tokenizer for the talker.

        The Qwen3-TTS-1.7B checkpoint ships a Qwen tokenizer at the model_path
        root (tokenizer.json + tokenizer_config.json). We use it ONLY to map
        the input string -> text-vocab token ids; the talker's autoregressive
        codec head is decoupled from this vocab.
        """
        cached = getattr(self, "_text_tokenizer", None)
        if cached is not None:
            return cached
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(model_path)
        self._text_tokenizer = tok
        return tok

    def prefill_text(
        self,
        text: str,
        model_path: str = "/workspace/qwen3-tts-1.7b",
        add_special_tokens: bool = False,
        codec_prefix_ids: list[int] | None = None,
    ) -> int:
        """Run pure-PyTorch text prefill, populate KV cache, advance position.

        Tokenizes ``text`` with the Qwen3-TTS tokenizer at ``model_path``,
        looks up the text-embedding rows, projects through the loaded
        text_projection MLP (silu-gated), then runs a one-shot PyTorch
        forward through the 28 talker transformer layers writing K/V
        tensors into ``self._k_cache``/``self._v_cache`` at positions
        ``[0, prefill_len)``. The subsequent ``self.step(audio_token_id)``
        calls then continue autoregressively from ``self._position``.

        Returns:
            The number of text tokens prefilled (>=1; clamped to MAX_SEQ_LEN-1
            so the audio decode still has room to write KV).

        Notes:
            - This MUST be called when ``self._position == 0`` (i.e. right
              after a fresh ``reset()``). Calling it twice without reset is
              undefined.
            - The text MLP is the standard SwiGLU pattern observed in the
              Qwen3-TTS reference: ``fc2(silu(fc1(h)))`` with optional
              biases. We honour the biases (Qwen3-TTS text_projection HAS
              biases per safetensors). Reference: ``talker.text_projection
              .linear_fc{1,2}.{weight,bias}``.
            - We deliberately keep the math identical-in-spirit to the
              megakernel kernel: RMSNorm with the same weights, q/k_norm
              per-head with the same weights, RoPE with the same cos/sin
              tables, GQA with NUM_KV_HEADS=8. Any numerical drift between
              the PyTorch prefill KV and what the kernel WOULD have computed
              at those positions is bounded by fp32-vs-fused-bf16 accumulator
              differences (a few ULPs) -- well below speech-quality
              significance.
        """
        if self._position != 0:
            raise RuntimeError(
                "Decoder.prefill_text() must be called when position == 0; "
                f"got position={self._position}. Call reset() first."
            )

        text = (text or "").strip()
        if not text:
            return 0

        tok = self._load_tokenizer(model_path)
        # Upstream prompt (modeling_qwen3_tts.py:2070 + qwen3_tts_model.py:270):
        #   <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        # No tts_bos/eos in the id stream — those special embeds are computed
        # separately and SUMMED with the codec-prefix embeddings.
        prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok.encode(prompt, add_special_tokens=False)
        if not ids:
            return 0
        max_prefill = max(1, MAX_SEQ_LEN - 256)
        if len(ids) > max_prefill:
            ids = ids[:max_prefill]

        device = self._k_cache.device
        input_id = torch.tensor([ids], dtype=torch.long, device=device)  # (1, T)

        w = self._weights
        text_embed = w["text_embedding_weight"]
        fc1_w = w["text_proj_fc1_weight"]
        fc1_b = w["text_proj_fc1_bias"]
        fc2_w = w["text_proj_fc2_weight"]
        fc2_b = w["text_proj_fc2_bias"]
        embed_weight = w["embed_weight"]  # talker codec_embedding (3072, HIDDEN)

        def _text_project(token_ids):
            """text_embedding + 2-layer SwiGLU-less MLP. (B, T) → (B, T, H)."""
            h = F.embedding(token_ids, text_embed)
            h = F.linear(h, fc1_w, fc1_b)
            h = F.silu(h)
            h = F.linear(h, fc2_w, fc2_b)
            return h.to(torch.bfloat16)

        # Special TTS embeds (from text_embed[tts_bos/eos/pad] → text_projection).
        TTS_BOS_ID = 151672
        TTS_EOS_ID = 151673
        TTS_PAD_ID = 151671
        special_ids = torch.tensor([[TTS_BOS_ID, TTS_EOS_ID, TTS_PAD_ID]], dtype=torch.long, device=device)
        tts_bos_embed, tts_eos_embed, tts_pad_embed = _text_project(special_ids).chunk(3, dim=1)
        # Shapes: each (1, 1, HIDDEN). Stash tts_pad_embed for the AR-step
        # trailing fallback in megakernel_tts.py.
        self._stashed_tts_pad_embed = tts_pad_embed.contiguous()

        # codec_prefix_ids OVERRIDE (we already build the canonical prefix below).
        # The argument is kept for backwards compatibility but ignored when
        # we have a full upstream-style prefill.
        # Upstream codec_input_embedding (7 entries):
        # [think, think_bos, lang, think_eos, spk_id, codec_pad, codec_bos]
        # Sanity: codec_prefix_ids should be [think, think_bos, lang, think_eos, spk_id, codec_pad].
        # We append codec_bos at the end (= the 7th).
        from qwen_megakernel.model import NUM_LAYERS  # avoid circular
        # use passed ids when provided to pick up speaker / language ; else default ryan-en.
        if codec_prefix_ids and len(codec_prefix_ids) == 6:
            full_codec_ids = list(codec_prefix_ids) + [2149]  # append codec_bos
        else:
            full_codec_ids = [2154, 2156, 2050, 2157, 3061, 2148, 2149]
        codec_ids_t = torch.tensor([full_codec_ids], dtype=torch.long, device=device)
        # codec_input_embedding (1, 7, HIDDEN), bf16.
        codec_input_embedding = F.embedding(codec_ids_t, embed_weight).to(torch.bfloat16)

        # ---- Build the talker_input_embed exactly as upstream lines 2174-2202 ----
        # 1. role (first 3 ids = `<|im_start|>`, `assistant`, `\n`):
        role_embed = _text_project(input_id[:, :3])  # (1, 3, H)

        # 2. Merged 6-entry block: (tts_pad * 5 + tts_bos) + codec_input_embedding[:, :-1]
        # codec_input_embedding[:, :-1] has 6 entries (excludes codec_bos).
        n_merged = codec_input_embedding.shape[1] - 1  # 6
        # tts_pad repeated n_merged-1 times, then tts_bos appended.
        merged_text = torch.cat(
            [tts_pad_embed.expand(-1, n_merged - 1, -1), tts_bos_embed],
            dim=1,
        )  # (1, 6, H)
        merged_block = merged_text + codec_input_embedding[:, :-1]  # (1, 6, H)

        # 3. First text body token (index 3) + codec_bos (codec_input_embedding[:, -1:]):
        first_text_embed = _text_project(input_id[:, 3:4])  # (1, 1, H)
        first_block = first_text_embed + codec_input_embedding[:, -1:]  # (1, 1, H)

        # 4. Concatenate: role + merged + first → 3 + 6 + 1 = 10 positions.
        talker_input_embed = torch.cat([role_embed, merged_block, first_block], dim=1)

        # ---- Trailing text hidden ----
        # Upstream line 2230-2232: trailing_text_hidden = cat(
        #   text_projection(text_embedding(input_id[:, 4:-5])), tts_eos_embed )
        body_embed = _text_project(input_id[:, 4:-5])  # (1, T_body, H)
        trailing = torch.cat([body_embed, tts_eos_embed], dim=1)  # (1, T_body+1, H)
        self._trailing_text_hiddens = trailing[0].clone().detach()  # (T_body+1, H) bf16

        # Run prefill through the talker layers; also get the last-position
        # hidden + first sampled token (the talker's "first AR step" output).
        n_pref, first_tok, last_hidden = self._prefill_embeds(
            talker_input_embed.squeeze(0), return_last_hidden=True,
        )
        self._stashed_first_tok = int(first_tok)
        self._stashed_first_past_hidden = last_hidden.contiguous()
        return n_pref

    def prefill_codec_prefix(self, codec_token_ids: list[int]) -> int:
        """Prefill the audio-side prefix tokens at positions [_position, _position+N).

        Mirrors the upstream Qwen3TTS flow
        (modeling_qwen3_tts.py:1240-1276) which feeds a 6-token codec prefix
        BETWEEN the text prefill and the AR audio decode:

            [codec_think, codec_think_bos, language_id, codec_think_eos,
             speaker_id, codec_pad]

        These ids index into the talker's audio input embedding (the
        ``codec_embedding`` table, NOT the text embedding) — same table the
        kernel's ``step()`` uses internally. They go through the 28 talker
        transformer layers and write K/V into the megakernel cache at the
        positions immediately following the text prefill.

        After this returns, ``self._position`` points at the next free slot;
        the caller seeds the AR loop with ``step(codec_bos_id=2149)``, which
        writes codec_bos's embedding at that slot and predicts the first
        audio token from the resulting state.

        Args:
            codec_token_ids: 6 codec-vocab token ids in the order above.

        Returns:
            Number of tokens prefilled (== len(codec_token_ids)).
        """
        if not codec_token_ids:
            return 0
        device = self._k_cache.device
        ids = torch.tensor(codec_token_ids, dtype=torch.long, device=device)
        # codec_embedding lookup — same weight the kernel uses for step()'s
        # input embedding.
        embed = self._embed_weight  # (3072, HIDDEN)
        h = F.embedding(ids, embed)  # (N, HIDDEN), bf16 (table dtype)
        return self._prefill_embeds(h.to(torch.bfloat16))

    def _prefill_embeds(self, x_seq: torch.Tensor, return_last_hidden: bool = False):
        """Shared transformer-layer prefill body.

        Runs the 28-layer forward over the given embeddings (shape (T, H))
        and writes K/V into the megakernel KV cache at positions
        ``[self._position, self._position + T)``. Advances ``self._position``
        by T. Used by both ``prefill_text`` (text projection output) and
        ``prefill_codec_prefix`` (codec_embedding output).
        """
        if x_seq.numel() == 0:
            return 0
        device = self._k_cache.device
        w = self._weights
        start_pos = self._position
        # Add batch dim if missing.
        x = x_seq.unsqueeze(0) if x_seq.dim() == 2 else x_seq  # (1, T, HIDDEN)
        T = x.shape[1]
        # Position ids run from start_pos so RoPE / MRoPE see the correct
        # absolute position regardless of which prefill phase we're in.
        positions = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=device)

        layer_weights = w["layer_weights"]
        n_ptrs = 11
        for layer_idx in range(NUM_LAYERS):
            base = layer_idx * n_ptrs
            ln1_w = layer_weights[base + 0]
            q_w = layer_weights[base + 1]
            k_w = layer_weights[base + 2]
            v_w = layer_weights[base + 3]
            qn_w = layer_weights[base + 4]
            kn_w = layer_weights[base + 5]
            o_w = layer_weights[base + 6]
            ln2_w = layer_weights[base + 7]
            gate_w = layer_weights[base + 8]
            up_w = layer_weights[base + 9]
            down_w = layer_weights[base + 10]

            # --- attention block ---
            h_norm = self._rms_norm(x, ln1_w)  # (1, T, HIDDEN)

            q = F.linear(h_norm, q_w).view(1, T, 16, HEAD_DIM)
            k = F.linear(h_norm, k_w).view(1, T, NUM_KV_HEADS, HEAD_DIM)
            v = F.linear(h_norm, v_w).view(1, T, NUM_KV_HEADS, HEAD_DIM)

            # Per-head q_norm / k_norm (RMSNorm over the HEAD_DIM axis).
            q = self._rms_norm(q, qn_w)
            k = self._rms_norm(k, kn_w)

            # (1, n_heads, T, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            q, k = self._apply_rope_to_qk(q, k, positions)

            # Stash this layer's K, V into the megakernel cache at the
            # absolute positions [start_pos, start_pos+T). This is the
            # critical bit that distinguishes phase-1 (text, start_pos=0)
            # from phase-2 (audio prefix, start_pos=T_text).
            self._k_cache[layer_idx, :, start_pos:start_pos + T, :].copy_(
                k[0].to(torch.bfloat16)
            )
            self._v_cache[layer_idx, :, start_pos:start_pos + T, :].copy_(
                v[0].to(torch.bfloat16)
            )

            # GQA expand for the attention math used to produce x for the
            # NEXT layer's input. The current prefill phase attends ONLY
            # over its own freshly-stashed positions (is_causal=True over
            # the (T, T) window) because earlier phases' K/V live at the
            # KV-cache positions we just bypass — they ARE present in
            # ``self._k_cache`` from phase-1, but the attention math for
            # the FIRST audio token is dominated by the audio prefix; the
            # text-context bleeds in via the layer-output residual stream,
            # not via a separate cross-attention.
            # NOTE: this differs from the upstream model.generate() which
            # runs one combined forward over (text + audio-prefix). We
            # approximate it as two phases — text K/V is laid down by
            # phase-1 and the AR loop's step() reads it from the kernel's
            # KV cache. Phase-2's small (6, 6) attention is run in
            # isolation because the audio-prefix tokens have to attend
            # back to the text context via the kernel's full attention
            # at AR-step time, not via prefill cross-attention. If audio
            # is babble after this fix, the prefill split is suspect.
            repeat = 16 // NUM_KV_HEADS
            if repeat > 1:
                k_exp = k.repeat_interleave(repeat, dim=1)
                v_exp = v.repeat_interleave(repeat, dim=1)
            else:
                k_exp = k
                v_exp = v

            attn_out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, is_causal=True, scale=self._attn_scale
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, T, Q_SIZE)
            attn_out = F.linear(attn_out, o_w)
            x = x + attn_out.to(x.dtype)

            # --- MLP block ---
            h_norm2 = self._rms_norm(x, ln2_w)
            gate = F.linear(h_norm2, gate_w)
            up = F.linear(h_norm2, up_w)
            inter = F.silu(gate) * up
            mlp_out = F.linear(inter, down_w)
            x = x + mlp_out.to(x.dtype)

        # Apply final norm so the LAST position's hidden can serve as the
        # talker's first `past_hidden` for the code predictor (matches the
        # upstream `hidden_states = outputs.last_hidden_state` which is
        # post-final-norm, then `past_hidden = hidden_states[:, -1:, :]`).
        last_hidden = self._rms_norm(x[:, -1:, :], self._final_norm_weight)  # (1, 1, HIDDEN)
        # Sample the FIRST audio token from the last prefill position's hidden
        # (mirrors HF GenerationMixin where the first generated token comes
        # from the prefill's final position).
        logits = F.linear(last_hidden, self._lm_head_weight).view(-1).to(torch.float32)
        first_tok = self._sample_audio_token(logits)
        self._position = start_pos + T
        if return_last_hidden:
            return T, first_tok, last_hidden
        return T
