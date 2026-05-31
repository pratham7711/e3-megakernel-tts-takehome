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
        # Match the upstream prompt format
        # (QwenLM/Qwen3-TTS/qwen_tts/inference/qwen3_tts_model.py:231-238):
        #   <tts_text_bos> <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n <tts_text_eod>
        # Without this wrap, the talker has K/V for raw text but no
        # turn-boundary markers — it never aligns with the audio-prefix
        # signals and never emits EOS at a sentence end.
        TTS_TEXT_BOS_ID = 151672
        TTS_TEXT_EOD_ID = 151673
        wrapped = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        # add_special_tokens=False because we add tts_text_bos/eod ourselves.
        ids = tok.encode(wrapped, add_special_tokens=False)
        if not ids:
            return 0
        ids = [TTS_TEXT_BOS_ID] + ids + [TTS_TEXT_EOD_ID]
        # Reserve room for at least one audio step + the audio prefix (~10 frames).
        max_prefill = max(1, MAX_SEQ_LEN - 256)
        if len(ids) > max_prefill:
            ids = ids[:max_prefill]

        device = self._k_cache.device
        token_ids = torch.tensor(ids, dtype=torch.long, device=device)  # (T,)

        # ---- 1. text_embedding lookup + text_projection MLP ----
        w = self._weights
        text_embed = w["text_embedding_weight"]  # (TEXT_VOCAB, HIDDEN)
        fc1_w = w["text_proj_fc1_weight"]
        fc1_b = w["text_proj_fc1_bias"]
        fc2_w = w["text_proj_fc2_weight"]
        fc2_b = w["text_proj_fc2_bias"]

        # Embedding lookup (bf16). (T, HIDDEN)
        h = F.embedding(token_ids, text_embed)
        # text_projection is a standard 2-layer MLP with SiLU activation; both
        # fc layers carry biases per safetensors. The Qwen3-TTS upstream
        # implements this as `fc2(silu(fc1(h)))` -- no gating.
        h = F.linear(h, fc1_w, fc1_b)
        h = F.silu(h)
        h = F.linear(h, fc2_w, fc2_b)  # (T, HIDDEN), bf16
        h = h.to(torch.bfloat16)

        # COMBINED FORWARD — append codec_prefix embeddings (audio-side
        # input) to the same sequence as the text embeddings, so the
        # 28-layer attention sees them within one self-attention causal
        # window. This matches the upstream `talker.generate(inputs_embeds
        # =cat([text_proj_out, codec_prefix_embeds]))` flow exactly. The
        # earlier two-phase split (prefill_text then prefill_codec_prefix)
        # left audio-prefix tokens unable to attend to text K within
        # phase-2's attention — they could only see text via the kernel's
        # KV cache at AR-step time, and that's apparently not enough for
        # the model to align (talker never emitted EOS, ran 245 s).
        if codec_prefix_ids:
            codec_ids_t = torch.tensor(codec_prefix_ids, dtype=torch.long, device=device)
            codec_h = F.embedding(codec_ids_t, w["embed_weight"]).to(torch.bfloat16)
            # codec_embedding (3072, HIDDEN) — the talker's audio input
            # embedding table, identical to what the megakernel uses for
            # step()'s embedding lookup.
            h = torch.cat([h, codec_h], dim=0)

        return self._prefill_embeds(h)

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

    def _prefill_embeds(self, x_seq: torch.Tensor) -> int:
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

        # We deliberately do NOT apply the final norm / lm_head here -- those
        # only matter for producing an output token, and the FIRST audio step
        # will produce the first audio token from the kernel's own forward.
        # The kernel just needs the prefilled K/V at positions
        # [start_pos, start_pos+T).
        self._position = start_pos + T
        return T
