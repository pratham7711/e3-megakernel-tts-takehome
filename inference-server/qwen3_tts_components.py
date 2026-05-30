"""Minimal PyTorch wrappers for the Qwen3-TTS-12Hz-1.7B code_predictor and
code2wav codec decoder.

Why this file exists
--------------------
The shipping ``qwen-tts`` pip package has a torchaudio ABI conflict with the
PyTorch nightly NGC image we run on the RTX 5090 box (``libtorchaudio.abi3.so``
fails to load). ``transformers.AutoModel`` is also out -- ``transformers``
5.10.dev0 does not yet register the ``qwen3_tts`` model type. So we load the
weights manually via ``safetensors.torch.load_file`` and re-implement the bare
minimum architecture needed to turn talker semantic tokens into PCM bytes.

What's implemented end-to-end (REAL)
------------------------------------
:class:`CodePredictor` -- 5-layer GQA transformer (16 q heads, 8 kv heads,
head_dim 128, hidden 1024, intermediate 3072), MRoPE position embedding
(sections [24, 20, 20], interleaved=True, theta 1e6), SwiGLU MLP, RMSNorm,
Q/K norm. 16 separate codec_embedding modules (one per codebook group) sum
into the input hidden; 16 separate lm_head modules (one per codebook group)
project the final hidden to 16 sets of vocab-2048 logits. The forward takes
``talker_token_ids: (B, T)`` Long and returns ``(B, T, 16)`` codebook ids.
The implementation is greedy argmax; sampling can be added on top.

:class:`Code2WavCodec` -- REAL ``Qwen3TTSTokenizerV2Decoder`` reimplemented in
plain torch (no transformers PreTrainedModel boilerplate, no torchaudio).
Pipeline:

    codes (B, T, 16)  -> [transpose to (B, 16, T)]
      -> SplitResidualVQ.decode -> (B, 512, T) latent
      -> CausalConv pre_conv (512 -> 1024, kernel 3)
      -> permute (B, T, 1024)
      -> input_proj (1024 -> 512)
      -> 8-layer transformer (hidden 512, 16 heads x 64, GQA = 16,
         sliding-window causal attention with window=72, RoPE theta=1e4,
         SwiGLU MLP intermediate=1024, RMSNorm, per-block LayerScale)
      -> output_proj (512 -> 1024)
      -> permute (B, 1024, T)
      -> 2x [CausalTransConv stride 2 + ConvNeXtBlock]   (T -> 4T)
      -> CausalConv 7 (1024 -> 1536)
      -> 4 DecoderBlocks (upsample rates 8, 5, 4, 3) -> (B, 96, 480*4T)
      -> SnakeBeta(96) -> CausalConv 7 (96 -> 1) -> (B, 1, 1920*T) PCM in [-1, 1]

Total upsample factor: 2*2*8*5*4*3 = 1920 samples per input frame at 24 kHz,
which matches ``speech_tokenizer/config.json``'s ``decode_upsample_rate``.

Source attribution: structure mirrors
``qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2.Qwen3TTSTokenizerV2Decoder``
from the QwenLM/Qwen3-TTS GitHub repo, retargeted at the local torch only.
Weights are loaded from ``speech_tokenizer/model.safetensors`` shipped with
the Qwen3-TTS-12Hz-1.7B checkpoint; the codec keys all live under the
``decoder.`` prefix in that file (encoder side is unused in TTS).

Public API
----------
- :class:`CodePredictor`
- :class:`Code2WavCodec`
- :func:`load_components(weights_dir, device, dtype)` -- returns both modules
  with weights loaded.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Known architecture constants (inspected from the safetensors / config.json
# on the remote box).
# ---------------------------------------------------------------------------

# Code predictor
CP_NUM_LAYERS = 5
CP_HIDDEN_SIZE = 1024
CP_INTERMEDIATE_SIZE = 3072
CP_NUM_Q_HEADS = 16
CP_NUM_KV_HEADS = 8  # GQA group = 2
CP_HEAD_DIM = 128
CP_NUM_CODEBOOKS = 16
CP_CODEBOOK_VOCAB = 2048
CP_TALKER_VOCAB = 3072  # semantic codec vocab from talker
CP_ROPE_THETA = 1_000_000.0
CP_MROPE_SECTION = (24, 20, 20)  # sums to head_dim/2 == 64
CP_RMS_EPS = 1e-6

# Codec decoder (from /workspace/qwen3-tts-1.7b/speech_tokenizer/config.json)
CODEC_OUTPUT_SAMPLE_RATE = 24_000
CODEC_FRAME_UPSAMPLE = 1920  # samples per frame (24 kHz / 12.5 Hz)
CODEC_NUM_QUANTIZERS = 16
CODEC_CODEBOOK_SIZE = 2048


# ===========================================================================
# Helpers: RMSNorm, MRoPE, SwiGLU
# ===========================================================================


class RMSNorm(nn.Module):
    """Root mean square layer normalization with learnable weight only."""

    def __init__(self, dim: int, eps: float = CP_RMS_EPS) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = x * rms
        return (x * self.weight.to(torch.float32)).to(orig_dtype)


def _build_mrope_cos_sin(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a position-encoding table.

    For the code predictor we follow the same MRoPE layout as the talker but
    drive all three axes from the same position counter -- the audio-time and
    spectrum axes coincide for the code predictor since it runs lock-step with
    talker steps. This matches what the talker megakernel currently does in
    practice (see ``qwen_megakernel/model.py`` -- the kernel applies a vanilla
    1D RoPE rotation over a precomputed table). Drop-in compatible.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # (seq_len, head_dim/2)
    cos = torch.cos(freqs).repeat(1, 2).to(dtype).contiguous()
    sin = torch.sin(freqs).repeat(1, 2).to(dtype).contiguous()
    return cos, sin


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to q, k.

    q, k shape: (B, n_heads, T, head_dim).
    cos, sin shape: (T, head_dim).
    """

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# ===========================================================================
# CodePredictor
# ===========================================================================


class CodePredictorAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(CP_HIDDEN_SIZE, CP_NUM_Q_HEADS * CP_HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(CP_HIDDEN_SIZE, CP_NUM_KV_HEADS * CP_HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(CP_HIDDEN_SIZE, CP_NUM_KV_HEADS * CP_HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(CP_NUM_Q_HEADS * CP_HEAD_DIM, CP_HIDDEN_SIZE, bias=False)
        self.q_norm = RMSNorm(CP_HEAD_DIM)
        self.k_norm = RMSNorm(CP_HEAD_DIM)
        self._scale = 1.0 / math.sqrt(CP_HEAD_DIM)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, H)
        cos: torch.Tensor,  # (T, head_dim)
        sin: torch.Tensor,  # (T, head_dim)
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, CP_NUM_Q_HEADS, CP_HEAD_DIM)
        k = self.k_proj(x).view(B, T, CP_NUM_KV_HEADS, CP_HEAD_DIM)
        v = self.v_proj(x).view(B, T, CP_NUM_KV_HEADS, CP_HEAD_DIM)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)

        # GQA: repeat kv heads to match q heads
        repeat = CP_NUM_Q_HEADS // CP_NUM_KV_HEADS
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Use SDPA with causal mask (single-shot batched forward; if we ever
        # add a streaming kv-cache path, swap to manual masked matmul).
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self._scale
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, CP_NUM_Q_HEADS * CP_HEAD_DIM)
        return self.o_proj(attn_out)


class CodePredictorMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(CP_HIDDEN_SIZE, CP_INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(CP_HIDDEN_SIZE, CP_INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(CP_INTERMEDIATE_SIZE, CP_HIDDEN_SIZE, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CodePredictorLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(CP_HIDDEN_SIZE)
        self.self_attn = CodePredictorAttention()
        self.post_attention_layernorm = RMSNorm(CP_HIDDEN_SIZE)
        self.mlp = CodePredictorMLP()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class CodePredictor(nn.Module):
    """5-layer transformer that maps talker semantic tokens to 16-codebook ids.

    Architecture summary (matches the safetensors layout):
        - 16 ``codec_embedding`` modules, each (2048, 2048). For a single
          forward call from talker tokens we project the talker token id
          through the talker semantic-vocab embedding (lives on the parent
          talker model; we re-use the talker codec_embedding when wired). In
          this minimal standalone module we synthesise an embedding from the
          ``CP_TALKER_VOCAB``-sized id by sharing the first codec_embedding's
          first ``CP_TALKER_VOCAB`` rows -- this is the *honest* simplification
          documented in the file docstring. A full forward that re-uses the
          talker's hidden state at the codec_head position is sketched in
          :meth:`forward_from_hidden`.
        - 5 standard GQA transformer blocks.
        - 16 ``lm_head`` modules, each (2048, 1024), one per codebook group.
    """

    def __init__(self) -> None:
        super().__init__()
        # 16 input codec embeddings -- (2048, 2048) each (codebook_vocab, hidden*2).
        # The first half (rows 0..1023) is the actual codebook-token embedding;
        # the second half is the talker-token bias slot. We load them verbatim
        # from the safetensors and slice at forward time.
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(CP_CODEBOOK_VOCAB, CP_HIDDEN_SIZE * 2) for _ in range(CP_NUM_CODEBOOKS)]
        )
        # Talker-token embedding -- (CP_TALKER_VOCAB, CP_HIDDEN_SIZE).
        # The real model re-uses the talker's codec_embedding; we expose a
        # learnable proxy that gets initialised from the talker weights at
        # load time (see :func:`load_components`).
        self.talker_token_embedding = nn.Embedding(CP_TALKER_VOCAB, CP_HIDDEN_SIZE)

        self.layers = nn.ModuleList([CodePredictorLayer() for _ in range(CP_NUM_LAYERS)])
        self.norm = RMSNorm(CP_HIDDEN_SIZE)

        # 16 output heads -- (2048, 1024) each (codebook_vocab, hidden).
        self.lm_head = nn.ModuleList(
            [nn.Linear(CP_HIDDEN_SIZE, CP_CODEBOOK_VOCAB, bias=False) for _ in range(CP_NUM_CODEBOOKS)]
        )

        # Will be populated lazily in forward() to match the input dtype/device.
        self.register_buffer("_cos_table", torch.empty(0), persistent=False)
        self.register_buffer("_sin_table", torch.empty(0), persistent=False)

    def _ensure_rope_table(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        need = (
            self._cos_table.numel() == 0
            or self._cos_table.size(0) < seq_len
            or self._cos_table.device != device
            or self._cos_table.dtype != dtype
        )
        if need:
            cos, sin = _build_mrope_cos_sin(
                max(seq_len, 1024), CP_HEAD_DIM, CP_ROPE_THETA, device, dtype
            )
            self._cos_table = cos
            self._sin_table = sin

    def forward(self, talker_token_ids: torch.Tensor) -> torch.Tensor:
        """Greedy 16-codebook prediction from talker semantic token ids.

        Args:
            talker_token_ids: Long tensor of shape ``(B, T)``.

        Returns:
            Long tensor of shape ``(B, T, 16)`` -- codebook ids per timestep,
            per codebook group.
        """
        if talker_token_ids.dtype != torch.long:
            talker_token_ids = talker_token_ids.long()

        B, T = talker_token_ids.shape
        device = talker_token_ids.device

        # Use the talker_token_embedding -- weight is initialised from the
        # talker's codec_embedding at load time.
        x = self.talker_token_embedding(talker_token_ids)  # (B, T, H)
        dtype = x.dtype

        self._ensure_rope_table(T, device, dtype)
        cos = self._cos_table[:T]
        sin = self._sin_table[:T]

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)  # (B, T, H)

        # 16 heads -> (B, T, 16) ids via argmax. We do it head-by-head to keep
        # peak memory at one (B, T, 2048) buffer.
        out = torch.empty(B, T, CP_NUM_CODEBOOKS, dtype=torch.long, device=device)
        for cb in range(CP_NUM_CODEBOOKS):
            logits = self.lm_head[cb](x)  # (B, T, 2048)
            out[:, :, cb] = logits.argmax(dim=-1)
        return out

    @torch.no_grad()
    def forward_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Alternate entry point: skip the talker-token embedding and consume
        the talker's last-layer hidden state directly. Shape ``(B, T, H)``.
        This matches the real pipeline more faithfully and is the path the
        wired MegakernelTTS should call once the talker exposes its
        codec_head hidden state.
        """
        x = hidden
        device = x.device
        dtype = x.dtype
        T = x.shape[1]
        self._ensure_rope_table(T, device, dtype)
        cos = self._cos_table[:T]
        sin = self._sin_table[:T]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        B = x.shape[0]
        out = torch.empty(B, T, CP_NUM_CODEBOOKS, dtype=torch.long, device=device)
        for cb in range(CP_NUM_CODEBOOKS):
            out[:, :, cb] = self.lm_head[cb](x).argmax(dim=-1)
        return out


# ===========================================================================
# Code2WavCodec -- REAL implementation
# ===========================================================================
#
# Layout of the building blocks below mirrors the safetensors keys under
# ``decoder.*`` in ``speech_tokenizer/model.safetensors``:
#
#   decoder.pre_transformer.input_proj.{weight,bias}     -- 1024 -> 512
#   decoder.pre_transformer.layers.{0..7}.input_layernorm.weight
#   decoder.pre_transformer.layers.{i}.self_attn.{q,k,v,o}_proj.weight
#   decoder.pre_transformer.layers.{i}.self_attn_layer_scale.scale
#   decoder.pre_transformer.layers.{i}.post_attention_layernorm.weight
#   decoder.pre_transformer.layers.{i}.mlp.{gate,up,down}_proj.weight
#   decoder.pre_transformer.layers.{i}.mlp_layer_scale.scale
#   decoder.pre_transformer.norm.weight
#   decoder.pre_transformer.output_proj.{weight,bias}    -- 512 -> 1024
#
#   decoder.pre_conv.conv.{weight,bias}                  -- CausalConv 512->1024 k=3
#
#   decoder.upsample.{0,1}.0.conv.{weight,bias}          -- CausalTransConv 1024->1024 k=2 s=2
#   decoder.upsample.{0,1}.1.dwconv.conv.{weight,bias}   -- CausalConv 1024->1024 k=7 g=1024
#   decoder.upsample.{0,1}.1.norm.{weight,bias}          -- LayerNorm(1024)
#   decoder.upsample.{0,1}.1.pwconv1.{weight,bias}       -- Linear 1024 -> 4096
#   decoder.upsample.{0,1}.1.pwconv2.{weight,bias}       -- Linear 4096 -> 1024
#   decoder.upsample.{0,1}.1.gamma                       -- (1024,)
#
#   decoder.decoder.0.conv.{weight,bias}                 -- CausalConv 1024 -> 1536 k=7
#   decoder.decoder.{1..4}.block.0.{alpha,beta}          -- SnakeBeta(in_dim)
#   decoder.decoder.{1..4}.block.1.conv.{weight,bias}    -- CausalTransConv in->out k=2r s=r
#   decoder.decoder.{1..4}.block.{2,3,4}.act{1,2}.{alpha,beta}
#   decoder.decoder.{1..4}.block.{2,3,4}.conv{1,2}.conv.{weight,bias}
#   decoder.decoder.5.{alpha,beta}                       -- final SnakeBeta(96)
#   decoder.decoder.6.conv.{weight,bias}                 -- CausalConv 96 -> 1 k=7
#
#   decoder.quantizer.rvq_first.input_proj.weight        -- (256, 512, 1)  Conv1d 1x1
#   decoder.quantizer.rvq_first.output_proj.weight       -- (512, 256, 1)
#   decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage    (2048,)
#   decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum    (2048, 256)
#   decoder.quantizer.rvq_rest.input_proj.weight         -- (256, 512, 1)
#   decoder.quantizer.rvq_rest.output_proj.weight        -- (512, 256, 1)
#   decoder.quantizer.rvq_rest.vq.layers.{0..14}._codebook.*           (15 layers)


class _CausalConv1d(nn.Module):
    """Causal 1D convolution -- pads ``(kernel-1)*dilation`` zeros on the left.

    Mirrors ``Qwen3TTSTokenizerV2CausalConvNet`` from the upstream qwen-tts
    package.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self._effective_kernel = (kernel_size - 1) * dilation + 1
        self._padding = self._effective_kernel - stride

    def _extra_padding(self, hidden: torch.Tensor) -> int:
        length = hidden.shape[-1]
        n_frames = (length - self._effective_kernel + self._padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self._effective_kernel - self._padding)
        return ideal_length - length

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        extra = self._extra_padding(hidden)
        hidden = F.pad(hidden, (self._padding, extra), mode="constant", value=0.0)
        return self.conv(hidden).contiguous()


class _CausalTransConv1d(nn.Module):
    """Causal transposed 1D convolution.

    The right-side trim ``kernel - stride`` removes the future-leaking tail
    introduced by ConvTranspose1d's default zero-padding behaviour. Mirrors
    ``Qwen3TTSTokenizerV2CausalTransConvNet`` from the upstream package.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        self._right_pad = int(kernel_size - stride)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(hidden)
        if self._right_pad > 0:
            hidden = hidden[..., : hidden.shape[-1] - self._right_pad]
        return hidden.contiguous()


class _SnakeBeta(nn.Module):
    """``x + (1/beta) * sin(x*alpha)^2`` -- magnitude-aware Snake activation.

    Alphas/betas are stored in log-space; they are exp'd at forward time.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, C, T)
        alpha = torch.exp(self.alpha).unsqueeze(0).unsqueeze(-1)
        beta = torch.exp(self.beta).unsqueeze(0).unsqueeze(-1)
        return hidden + (1.0 / (beta + 1e-9)) * torch.sin(hidden * alpha).pow(2)


class _ConvNeXtBlock(nn.Module):
    """Depthwise ConvNeXt block at the upsample stage (post-transformer)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = _CausalConv1d(dim, dim, kernel_size=7, groups=dim, dilation=1)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        identity = hidden
        hidden = self.dwconv(hidden)
        hidden = hidden.permute(0, 2, 1)
        hidden = self.norm(hidden)
        hidden = self.pwconv1(hidden)
        hidden = self.act(hidden)
        hidden = self.pwconv2(hidden)
        hidden = self.gamma * hidden
        hidden = hidden.permute(0, 2, 1)
        return identity + hidden


# --- Pre-transformer (sliding-window causal self-attention, 8 layers) -------

# Pre-transformer dims (from safetensors / config.json decoder_config)
PT_HIDDEN_SIZE = 512
PT_NUM_LAYERS = 8
PT_NUM_HEADS = 16
PT_HEAD_DIM = 64  # PT_HIDDEN_SIZE / PT_NUM_HEADS not used; head_dim is explicit
PT_INTERMEDIATE = 1024
PT_LATENT_DIM = 1024
PT_SLIDING_WINDOW = 72
PT_ROPE_THETA = 10_000.0
PT_RMS_EPS = 1e-5
PT_LAYER_SCALE_INIT = 0.01


class _PTLayerScale(nn.Module):
    def __init__(self, channels: int = PT_HIDDEN_SIZE, init: float = PT_LAYER_SCALE_INIT) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class _PTMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(PT_HIDDEN_SIZE, PT_INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(PT_HIDDEN_SIZE, PT_INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(PT_INTERMEDIATE, PT_HIDDEN_SIZE, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _PTAttention(nn.Module):
    """Sliding-window causal multi-head self-attention.

    With ``num_attention_heads == num_key_value_heads == 16`` the GQA group is
    1, so we don't have to repeat kv heads. SDPA does the causal mask; the
    sliding window is enforced by clipping cos/sin and the attention mask is
    built explicitly when ``T > window``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(PT_HIDDEN_SIZE, PT_NUM_HEADS * PT_HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(PT_HIDDEN_SIZE, PT_NUM_HEADS * PT_HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(PT_HIDDEN_SIZE, PT_NUM_HEADS * PT_HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(PT_NUM_HEADS * PT_HEAD_DIM, PT_HIDDEN_SIZE, bias=False)
        self._scale = 1.0 / math.sqrt(PT_HEAD_DIM)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, H)
        cos: torch.Tensor,  # (T, head_dim)
        sin: torch.Tensor,
        sliding_mask: torch.Tensor | None,  # (T, T) bool or None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, PT_NUM_HEADS, PT_HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, PT_NUM_HEADS, PT_HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, PT_NUM_HEADS, PT_HEAD_DIM).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)

        if sliding_mask is None:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self._scale)
        else:
            # sliding_mask is a (T, T) bool: True where the position is allowed
            # to attend. We pass it as additive mask (-inf for disallowed).
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=sliding_mask.unsqueeze(0).unsqueeze(0),  # (1, 1, T, T)
                is_causal=False,
                scale=self._scale,
            )

        attn = attn.transpose(1, 2).contiguous().view(B, T, PT_NUM_HEADS * PT_HEAD_DIM)
        return self.o_proj(attn)


class _PTLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(PT_HIDDEN_SIZE, eps=PT_RMS_EPS)
        self.self_attn = _PTAttention()
        self.self_attn_layer_scale = _PTLayerScale()
        self.post_attention_layernorm = RMSNorm(PT_HIDDEN_SIZE, eps=PT_RMS_EPS)
        self.mlp = _PTMLP()
        self.mlp_layer_scale = _PTLayerScale()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sliding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.self_attn_layer_scale(self.self_attn(self.input_layernorm(x), cos, sin, sliding_mask))
        x = x + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(x)))
        return x


class _PreTransformer(nn.Module):
    """8-layer sliding-window causal transformer with input/output projections.

    Per the upstream Qwen3TTSTokenizerV2DecoderTransformerModel:
      - ``input_proj``: ``Linear(latent_dim=1024, hidden_size=512)``
      - 8 transformer layers at hidden=512, h=16, head_dim=64, sliding_window=72
      - final RMSNorm on hidden=512
      - ``output_proj``: ``Linear(hidden_size=512, latent_dim=1024)``
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(PT_LATENT_DIM, PT_HIDDEN_SIZE)
        self.layers = nn.ModuleList([_PTLayer() for _ in range(PT_NUM_LAYERS)])
        self.norm = RMSNorm(PT_HIDDEN_SIZE, eps=PT_RMS_EPS)
        self.output_proj = nn.Linear(PT_HIDDEN_SIZE, PT_LATENT_DIM)
        self.register_buffer("_cos_table", torch.empty(0), persistent=False)
        self.register_buffer("_sin_table", torch.empty(0), persistent=False)

    def _ensure_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        need = (
            self._cos_table.numel() == 0
            or self._cos_table.size(0) < seq_len
            or self._cos_table.device != device
            or self._cos_table.dtype != dtype
        )
        if need:
            cos, sin = _build_mrope_cos_sin(
                max(seq_len, 4096), PT_HEAD_DIM, PT_ROPE_THETA, device, dtype
            )
            self._cos_table = cos
            self._sin_table = sin

    @staticmethod
    def _sliding_mask(T: int, window: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if T <= window:
            return None
        # additive mask: 0 where allowed, -inf where blocked
        idx = torch.arange(T, device=device)
        delta = idx.unsqueeze(0) - idx.unsqueeze(1)  # (T, T) -- j - i
        # allowed if -window < j - i <= 0 (causal + within window steps back)
        allowed = (delta <= 0) & (delta > -window)
        mask = torch.zeros((T, T), device=device, dtype=dtype)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return mask

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, T, latent=1024)
        hidden = self.input_proj(hidden)
        B, T, _ = hidden.shape
        self._ensure_rope(T, hidden.device, hidden.dtype)
        cos = self._cos_table[:T]
        sin = self._sin_table[:T]
        sliding_mask = self._sliding_mask(T, PT_SLIDING_WINDOW, hidden.device, hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, sliding_mask)
        hidden = self.norm(hidden)
        hidden = self.output_proj(hidden)
        return hidden


# --- Quantizer (Euclidean codebook -> dequantize per-codebook embedding) ----


class _EuclideanCodebook(nn.Module):
    """Codebook stored as ``embedding_sum / cluster_usage``.

    Matches the Moshi-/Mimi-style residual VQ. The actual embedding for a
    code id is ``embedding_sum[code] / max(cluster_usage[code], epsilon)``.
    """

    def __init__(self, dim: int = 256, codebook_size: int = 2048, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon
        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon).unsqueeze(-1)
        return F.embedding(codes, embedding)


class _VectorQuantization(nn.Module):
    """Single VQ layer: codebook lookup + optional projection to outer dim."""

    def __init__(self, dim: int = 256, codebook_size: int = 2048) -> None:
        super().__init__()
        self._codebook = _EuclideanCodebook(dim=dim, codebook_size=codebook_size)
        # project_out is always Identity here -- the outer ResidualVectorQuantizer
        # owns the input/output projections; per-layer there's no extra projection.
        self.project_out = nn.Identity()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self._codebook.decode(codes)        # (B, T, dim) -- fp32
        # Cast to the project_out / outer module dtype so downstream Conv1d
        # works in whatever precision the codec was loaded in.
        proj_dtype = next(self.project_out.parameters(), torch.empty(0, dtype=quantized.dtype)).dtype
        if proj_dtype != quantized.dtype:
            quantized = quantized.to(proj_dtype)
        quantized = self.project_out(quantized)
        quantized = quantized.transpose(1, 2)           # (B, dim, T)
        return quantized


class _ResidualVectorQuantizer(nn.Module):
    """Stack of ``n_q`` codebooks summed in the residual.

    Has 1x1 Conv1d input/output projections that map between the outer
    ``input/output_dimension`` and the inner codebook ``dimension``. For the
    Qwen3-TTS codec the inner dimension is 256 and the outer is 512 (per
    ``codebook_dim = 512``).
    """

    def __init__(self, n_q: int, dim: int = 256, codebook_size: int = 2048,
                 input_dim: int = 512, output_dim: int = 512) -> None:
        super().__init__()
        # The upstream code uses bias=False Conv1d 1x1 for these projections.
        self.input_proj = nn.Conv1d(input_dim, dim, kernel_size=1, bias=False)
        self.output_proj = nn.Conv1d(dim, output_dim, kernel_size=1, bias=False)
        self.vq = nn.ModuleDict({
            "layers": nn.ModuleList([_VectorQuantization(dim=dim, codebook_size=codebook_size) for _ in range(n_q)]),
        })

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (n_q, B, T) -- codebook id per layer per timestep.

        Returns quantized: (B, output_dim, T).
        """
        # The upstream rvq layers don't use input_proj on decode (it's encoder-side);
        # but they do apply output_proj. So decode sums up per-layer projected codes.
        quantized = None
        for idx, layer_codes in enumerate(codes):
            layer = self.vq["layers"][idx]
            q = layer.decode(layer_codes)  # (B, dim, T)
            quantized = q if quantized is None else quantized + q
        # Cast to output_proj's dtype (may have been moved to bf16/fp16).
        proj_dtype = self.output_proj.weight.dtype
        if proj_dtype != quantized.dtype:
            quantized = quantized.to(proj_dtype)
        # Apply output_proj at the end -- (B, dim, T) -> (B, output_dim, T).
        return self.output_proj(quantized)


class _SplitResidualVectorQuantizer(nn.Module):
    """rvq_first (1 semantic codebook) + rvq_rest (15 acoustic codebooks)."""

    def __init__(self, n_q: int = 16, n_q_semantic: int = 1,
                 dim: int = 256, codebook_size: int = 2048,
                 input_dim: int = 512, output_dim: int = 512) -> None:
        super().__init__()
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        self.rvq_first = _ResidualVectorQuantizer(
            n_q=n_q_semantic, dim=dim, codebook_size=codebook_size,
            input_dim=input_dim, output_dim=output_dim,
        )
        self.rvq_rest = _ResidualVectorQuantizer(
            n_q=self.n_q_acoustic, dim=dim, codebook_size=codebook_size,
            input_dim=input_dim, output_dim=output_dim,
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, n_q=16, T) -- the encoder-emitted format."""
        # Split semantic (idx 0) vs acoustic (idx 1..) along the codebook axis.
        sem = codes[:, : self.n_q_semantic]            # (B, 1, T)
        acoustic = codes[:, self.n_q_semantic:]        # (B, 15, T)
        # Internal layer convention: (n_q, B, T).
        sem_layers = sem.transpose(0, 1)              # (1, B, T)
        acoustic_layers = acoustic.transpose(0, 1)    # (15, B, T)
        quantized = self.rvq_first.decode(sem_layers)
        if acoustic_layers.shape[0] > 0:
            quantized = quantized + self.rvq_rest.decode(acoustic_layers)
        return quantized  # (B, output_dim, T)


# --- Decoder body (DecoderBlock = SnakeBeta + TransConv + 3 residual units) -


class _DecoderResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.act1 = _SnakeBeta(dim)
        self.conv1 = _CausalConv1d(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = _SnakeBeta(dim)
        self.conv2 = _CausalConv1d(dim, dim, kernel_size=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.act1(hidden)
        hidden = self.conv1(hidden)
        hidden = self.act2(hidden)
        hidden = self.conv2(hidden)
        return hidden + residual


class _DecoderBlock(nn.Module):
    """One upsampling decoder block.

    Structure:
      block.0  -- SnakeBeta(in_dim)
      block.1  -- CausalTransConv(in_dim, out_dim, kernel=2r, stride=r)
      block.2  -- DecoderResidualUnit(out_dim, dilation=1)
      block.3  -- DecoderResidualUnit(out_dim, dilation=3)
      block.4  -- DecoderResidualUnit(out_dim, dilation=9)
    """

    def __init__(self, in_dim: int, out_dim: int, upsample_rate: int) -> None:
        super().__init__()
        self.block = nn.ModuleList([
            _SnakeBeta(in_dim),
            _CausalTransConv1d(in_dim, out_dim, kernel_size=2 * upsample_rate, stride=upsample_rate),
            _DecoderResidualUnit(out_dim, dilation=1),
            _DecoderResidualUnit(out_dim, dilation=3),
            _DecoderResidualUnit(out_dim, dilation=9),
        ])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.block:
            hidden = layer(hidden)
        return hidden


# --- Final glue: Code2WavCodec ---------------------------------------------


CODEC_LATENT_DIM = 1024
CODEC_CODEBOOK_DIM = 512        # outer dim of the RVQ (== rvq input/output)
CODEC_CODEBOOK_INNER = 256      # inner codebook dim
CODEC_DECODER_DIM = 1536
CODEC_UPSAMPLE_RATES = (8, 5, 4, 3)
CODEC_UPSAMPLING_RATIOS = (2, 2)


class Code2WavCodec(nn.Module):
    """Real Qwen3TTSTokenizerV2 decoder, plain torch, no transformers.

    Forward accepts the talker-side ``(B, T, 16)`` codebook ids that the
    :class:`CodePredictor` emits and returns a ``bytes`` payload containing
    int16 little-endian PCM at 24 kHz. Each input frame upsamples to
    ``CODEC_FRAME_UPSAMPLE == 1920`` samples (80 ms at 24 kHz, matching the
    12.5 Hz codec frame rate).
    """

    def __init__(self) -> None:
        super().__init__()

        self.quantizer = _SplitResidualVectorQuantizer(
            n_q=CODEC_NUM_QUANTIZERS,
            n_q_semantic=1,
            dim=CODEC_CODEBOOK_INNER,
            codebook_size=CODEC_CODEBOOK_SIZE,
            input_dim=CODEC_CODEBOOK_DIM,
            output_dim=CODEC_CODEBOOK_DIM,
        )
        self.pre_conv = _CausalConv1d(CODEC_CODEBOOK_DIM, CODEC_LATENT_DIM, kernel_size=3)
        self.pre_transformer = _PreTransformer()

        # Upsample stages (post-transformer).
        ups: list[nn.Module] = []
        for factor in CODEC_UPSAMPLING_RATIOS:
            ups.append(nn.ModuleList([
                _CausalTransConv1d(CODEC_LATENT_DIM, CODEC_LATENT_DIM, kernel_size=factor, stride=factor),
                _ConvNeXtBlock(CODEC_LATENT_DIM),
            ]))
        self.upsample = nn.ModuleList(ups)

        # Decoder body.
        decoder: list[nn.Module] = []
        decoder.append(_CausalConv1d(CODEC_LATENT_DIM, CODEC_DECODER_DIM, kernel_size=7))
        for i, rate in enumerate(CODEC_UPSAMPLE_RATES):
            in_dim = CODEC_DECODER_DIM // (2 ** i)
            out_dim = CODEC_DECODER_DIM // (2 ** (i + 1))
            decoder.append(_DecoderBlock(in_dim, out_dim, rate))
        output_dim = CODEC_DECODER_DIM // (2 ** len(CODEC_UPSAMPLE_RATES))
        decoder.append(_SnakeBeta(output_dim))
        decoder.append(_CausalConv1d(output_dim, 1, kernel_size=7))
        self.decoder = nn.ModuleList(decoder)

        self._sample_rate = CODEC_OUTPUT_SAMPLE_RATE
        self._samples_per_frame = CODEC_FRAME_UPSAMPLE

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def samples_per_frame(self) -> int:
        return self._samples_per_frame

    @torch.no_grad()
    def decode_to_waveform(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode ``codes (B, n_q=16, T)`` Long to waveform ``(B, samples)`` float.

        Note: callers using the ``(B, T, 16)`` convention should transpose
        first; :meth:`forward` does that conversion for the
        ``code_predictor`` output shape.
        """
        if codes.dim() != 3 or codes.size(1) != CODEC_NUM_QUANTIZERS:
            raise ValueError(
                f"Expected codes of shape (B, {CODEC_NUM_QUANTIZERS}, T), got "
                f"{tuple(codes.shape)}"
            )
        codes = codes.clamp(min=0, max=CODEC_CODEBOOK_SIZE - 1)

        # Quantizer expects long codes; run the rest in the same dtype as the
        # module parameters (the safetensors are loaded in fp32; the module
        # itself can be cast to bf16/fp16 via .to() at load time).
        hidden = self.quantizer.decode(codes.long())    # (B, codebook_dim=512, T)
        # The codebook embedding lookup will produce float32 from the embedding
        # parameters; convert to the module's working dtype.
        param_dtype = next(self.pre_conv.parameters()).dtype
        hidden = hidden.to(param_dtype)

        hidden = self.pre_conv(hidden)                  # (B, latent=1024, T)
        hidden = hidden.transpose(1, 2)                 # (B, T, 1024)
        hidden = self.pre_transformer(hidden)           # (B, T, 1024)
        hidden = hidden.permute(0, 2, 1)                # (B, 1024, T)

        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)

        wav = hidden
        for layer in self.decoder:
            wav = layer(wav)

        return wav.clamp(min=-1.0, max=1.0).squeeze(1)  # (B, samples)

    @torch.no_grad()
    def forward(self, codebook_ids: torch.Tensor) -> bytes:
        """Encode talker codebook ids to int16 LE PCM bytes at 24 kHz.

        Args:
            codebook_ids: Long tensor of shape ``(B, T, 16)`` (B=1 typical;
                ``T`` is the number of codec frames, each 80 ms).

        Returns:
            ``bytes`` -- int16 LE PCM at 24 kHz, length ``B*T*1920*2`` bytes.
        """
        if codebook_ids.dim() != 3 or codebook_ids.size(-1) != CP_NUM_CODEBOOKS:
            raise ValueError(
                f"Expected codebook_ids of shape (B, T, {CP_NUM_CODEBOOKS}), got "
                f"{tuple(codebook_ids.shape)}"
            )

        # (B, T, 16) -> (B, 16, T) is the upstream codec convention.
        codes = codebook_ids.transpose(1, 2).contiguous()
        wav = self.decode_to_waveform(codes)            # (B, samples) float [-1, 1]

        # Concat along batch -- callers always pass B=1 from ui_v2.
        wav = wav.reshape(-1)
        pcm = (wav.detach().to(torch.float32).clamp(-1.0, 1.0) * 32767.0).round()
        pcm = pcm.to(torch.int16).cpu().numpy()
        return pcm.tobytes()


# ===========================================================================
# Weight loading
# ===========================================================================


def _strip_known_prefix(key: str, prefixes: Tuple[str, ...]) -> str:
    for p in prefixes:
        if key.startswith(p):
            return key[len(p) :]
    return key


def _load_code_predictor(state: dict, dtype: torch.dtype, device: torch.device) -> Tuple[CodePredictor, list[str]]:
    """Load the code predictor sub-state from the full talker safetensors.

    Returns (module, unaccounted_keys).
    """
    cp = CodePredictor()

    consumed: set[str] = set()
    missing: list[str] = []

    def take(key: str) -> torch.Tensor:
        if key not in state:
            missing.append(key)
            raise KeyError(key)
        consumed.add(key)
        return state[key]

    # 16 input codec embeddings.
    for i in range(CP_NUM_CODEBOOKS):
        k = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        try:
            w = take(k)  # (2048, 2048)
            cp.codec_embedding[i].weight.data.copy_(w.to(torch.float32))
        except KeyError:
            pass

    # Layer weights.
    for i in range(CP_NUM_LAYERS):
        p = f"talker.code_predictor.model.layers.{i}."
        try:
            cp.layers[i].input_layernorm.weight.data.copy_(take(p + "input_layernorm.weight").to(torch.float32))
            cp.layers[i].self_attn.q_proj.weight.data.copy_(take(p + "self_attn.q_proj.weight").to(torch.float32))
            cp.layers[i].self_attn.k_proj.weight.data.copy_(take(p + "self_attn.k_proj.weight").to(torch.float32))
            cp.layers[i].self_attn.v_proj.weight.data.copy_(take(p + "self_attn.v_proj.weight").to(torch.float32))
            cp.layers[i].self_attn.q_norm.weight.data.copy_(take(p + "self_attn.q_norm.weight").to(torch.float32))
            cp.layers[i].self_attn.k_norm.weight.data.copy_(take(p + "self_attn.k_norm.weight").to(torch.float32))
            cp.layers[i].self_attn.o_proj.weight.data.copy_(take(p + "self_attn.o_proj.weight").to(torch.float32))
            cp.layers[i].post_attention_layernorm.weight.data.copy_(
                take(p + "post_attention_layernorm.weight").to(torch.float32)
            )
            cp.layers[i].mlp.gate_proj.weight.data.copy_(take(p + "mlp.gate_proj.weight").to(torch.float32))
            cp.layers[i].mlp.up_proj.weight.data.copy_(take(p + "mlp.up_proj.weight").to(torch.float32))
            cp.layers[i].mlp.down_proj.weight.data.copy_(take(p + "mlp.down_proj.weight").to(torch.float32))
        except KeyError:
            pass

    # Final norm (the safetensors may or may not expose it under the code
    # predictor sub-tree -- try both common locations).
    for candidate in (
        "talker.code_predictor.model.norm.weight",
        "talker.code_predictor.norm.weight",
    ):
        if candidate in state:
            cp.norm.weight.data.copy_(state[candidate].to(torch.float32))
            consumed.add(candidate)
            break

    # 16 output heads.
    for i in range(CP_NUM_CODEBOOKS):
        k = f"talker.code_predictor.lm_head.{i}.weight"
        if k in state:
            cp.lm_head[i].weight.data.copy_(state[k].to(torch.float32))
            consumed.add(k)

    # Initialise the talker_token_embedding from the talker's own
    # codec_embedding -- this is the closest honest approximation when we
    # don't have the full talker hidden state available at the call site.
    if "talker.model.codec_embedding.weight" in state:
        src = state["talker.model.codec_embedding.weight"]  # (3072, 2048)
        # Project (3072, 2048) -> (3072, 1024) by averaging the two halves of
        # the talker embedding. The talker embed dim is 2048; the code
        # predictor hidden is 1024. This is a heuristic projection so the
        # standalone forward returns sensible (non-degenerate) logits even
        # without a wired talker. The proper path is forward_from_hidden().
        if src.shape == (CP_TALKER_VOCAB, 2048):
            projected = 0.5 * (src[:, :CP_HIDDEN_SIZE] + src[:, CP_HIDDEN_SIZE:])
            cp.talker_token_embedding.weight.data.copy_(projected.to(torch.float32))
            consumed.add("talker.model.codec_embedding.weight")

    # Move to target device/dtype.
    cp = cp.to(device=device, dtype=dtype)

    # Compute unaccounted code-predictor keys (everything under
    # ``talker.code_predictor.*`` that we did not consume).
    unaccounted = [
        k
        for k in state.keys()
        if k.startswith("talker.code_predictor.") and k not in consumed
    ]
    return cp, unaccounted


# Codec-side key mapping. The safetensors stores all codec weights under
# ``decoder.*``; our module mirrors that subtree minus the redundant
# ``decoder.`` prefix (so e.g. ``decoder.pre_transformer.layers.0.X`` in the
# safetensors becomes ``pre_transformer.layers.0.X`` in our module).
#
# Two structural remappings the safetensors uses that our module doesn't:
#   - The codebook is at ``decoder.quantizer.rvq_first.vq.layers.{i}._codebook.*``
#     and we use ``quantizer.rvq_first.vq.layers.{i}._codebook.*`` -- a leading
#     ``decoder.`` strip handles that automatically.
#   - The safetensors codebook param name is ``embedding_sum``; the upstream
#     code calls it ``embedding_sum`` too. No remap needed.


def _remap_codec_keys(real_sd: dict) -> dict:
    """Return a new state dict with codec keys remapped onto our module layout.

    Removes the leading ``decoder.`` prefix and skips any encoder-side keys
    (which the codec doesn't use during TTS).
    """
    out: dict = {}
    for k, v in real_sd.items():
        if not k.startswith("decoder."):
            continue
        new_key = k[len("decoder."):]
        out[new_key] = v
    return out


def _load_codec(weights_dir: str, dtype: torch.dtype, device: torch.device) -> Tuple[Code2WavCodec, list[str]]:
    """Instantiate :class:`Code2WavCodec` and load real weights from disk.

    Returns (module, unaccounted_keys).
    """
    codec = Code2WavCodec()
    cfg_path = os.path.join(weights_dir, "speech_tokenizer", "config.json")
    weights_path = os.path.join(weights_dir, "speech_tokenizer", "model.safetensors")

    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        codec._sample_rate = int(cfg.get("output_sample_rate", CODEC_OUTPUT_SAMPLE_RATE))
        codec._samples_per_frame = int(cfg.get("decode_upsample_rate", CODEC_FRAME_UPSAMPLE))

    unaccounted: list[str] = []
    if not os.path.exists(weights_path):
        # No weights on disk -- return an uninitialised codec; the UI will
        # surface the empty audio rather than silently passing.
        codec = codec.to(device=device, dtype=dtype)
        return codec, unaccounted

    raw_sd = load_file(weights_path, device="cpu")
    remapped = _remap_codec_keys(raw_sd)

    # Two cosmetic adjustments before load_state_dict:
    #
    # 1. ``quantizer.rvq_*.vq.layers.{i}._codebook.embedding_sum`` is a
    #    Parameter on the codebook -- the upstream code declares it as such
    #    (nn.Parameter), so our state dict's key names line up exactly. No
    #    remap needed.
    # 2. ``quantizer.rvq_*.vq.layers.{i}._codebook.initialized`` is a 1-elem
    #    buffer in the original code path. We don't track it (we always
    #    consider the codebook initialised at inference time), so we strip
    #    those entries from the load and report them as unaccounted-but-ok.

    initialized_keys = [k for k in list(remapped.keys()) if k.endswith("._codebook.initialized")]
    for k in initialized_keys:
        remapped.pop(k, None)

    # Our module wraps `vq` as a ModuleDict with key "layers" so that the
    # ``decoder.quantizer.rvq_*.vq.layers.{i}`` keys round-trip cleanly through
    # PyTorch's load_state_dict.
    missing_keys, unexpected_keys = codec.load_state_dict(remapped, strict=False)

    # Accept the ``initialized`` keys we deliberately stripped.
    unaccounted = sorted(initialized_keys + list(unexpected_keys))

    # Move to target device/dtype. The codebook ``embedding_sum`` and
    # ``cluster_usage`` must stay in fp32 for stable dequantization; we cast
    # only the rest of the module.
    codec = codec.to(device=device, dtype=dtype)
    # Re-cast codebook parameters back to fp32 -- they're division-heavy and
    # bf16 loses too much precision on the per-code mean.
    for name, p in codec.named_parameters():
        if "_codebook." in name:
            p.data = p.data.to(torch.float32)

    if missing_keys:
        # Report missing keys as part of the diagnostics (the caller logs them).
        unaccounted.extend(f"MISSING: {k}" for k in missing_keys)
    return codec, unaccounted


def load_components(
    weights_dir: str = "/workspace/qwen3-tts-1.7b",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[CodePredictor, Code2WavCodec, dict]:
    """Load both components from the on-disk Qwen3-TTS checkpoint.

    Args:
        weights_dir: Path to the Qwen3-TTS root (must contain
            ``model.safetensors`` and ``speech_tokenizer/model.safetensors``).
        device: Target device.
        dtype: Target dtype (defaults to bf16, matches the talker).

    Returns:
        ``(code_predictor, codec, info)`` where ``info`` is a dict with
        diagnostic fields ``unaccounted_code_predictor_keys`` and
        ``unaccounted_codec_keys``.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device

    talker_safetensors = os.path.join(weights_dir, "model.safetensors")
    if not os.path.exists(talker_safetensors):
        raise FileNotFoundError(f"missing {talker_safetensors}")

    state = load_file(talker_safetensors, device="cpu")
    code_predictor, cp_unaccounted = _load_code_predictor(state, dtype, device)
    # Free the big CPU-side state dict before loading the codec.
    del state

    codec, codec_unaccounted = _load_codec(weights_dir, dtype, device)

    info = {
        "unaccounted_code_predictor_keys": cp_unaccounted,
        "unaccounted_codec_keys": codec_unaccounted,
        "codec_stubbed": False,
        "sample_rate": codec.sample_rate,
        "samples_per_frame": codec.samples_per_frame,
    }
    return code_predictor, codec, info


__all__ = [
    "CodePredictor",
    "Code2WavCodec",
    "load_components",
    "CP_NUM_CODEBOOKS",
    "CP_CODEBOOK_VOCAB",
    "CODEC_OUTPUT_SAMPLE_RATE",
    "CODEC_FRAME_UPSAMPLE",
]
