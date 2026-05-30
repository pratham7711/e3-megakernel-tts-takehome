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

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position
