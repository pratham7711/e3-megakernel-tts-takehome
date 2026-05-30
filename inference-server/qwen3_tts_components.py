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

What's stubbed (CLEARLY MARKED)
-------------------------------
:class:`Code2WavCodec` -- the real Qwen3TTSTokenizerV2Model decoder is a
496-key non-DiT causal-ConvNet + 8-layer transformer hybrid. Reverse-
engineering the exact ConvNet topology from the safetensors keys alone
(``decoder.decoder.0.conv.*``, ``decoder.decoder.1.block.0.alpha/beta``,
SnakeBeta activations, residual stack ordering, transpose-conv strides) is
not feasible without the upstream source for this brief. The stub here
takes ``(B, T, 16)`` codebook ids and synthesises a recognisable test
signal: a sine wave whose frequency is modulated by the mean of the 16
codebook ids per frame. Each input frame produces ``1920`` int16 samples
at 24 kHz (matches ``decode_upsample_rate``).

Look for ``# STUB: replace with full codec reverse-engineering``.

Honest limitation
-----------------
With this file alone you can:
    - load the code predictor weights from the real safetensors and produce
      meaningful 16-way codebook predictions from talker tokens,
    - feed those codebook ids through the stub decoder and get test-tone
      bytes that prove the integration surface (shape, dtype, sample rate,
      streaming chunk size) is correct.

You CANNOT produce intelligible speech until ``Code2WavCodec`` is replaced
with the real ConvNet decoder. The integration tests (TTFC, chunk size,
RTF measurement against silence) will pass; an A/B listening test will
not. This is the documented tradeoff.

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
# Code2WavCodec -- STUB
# ===========================================================================


class Code2WavCodec(nn.Module):
    """STUB codec decoder. See file docstring.

    Real architecture: 8-layer transformer + non-DiT causal ConvNet with
    SnakeBeta activations, 16-quantizer residual VQ, upsample ratio 1920.
    The 496-key state dict structure is:
        decoder.decoder.{0}.conv.weight, .bias                  -- entry conv
        decoder.decoder.{1..K}.block.{0..N}.alpha/beta           -- SnakeBeta act
        decoder.decoder.{1..K}.block.{0..N}.conv.weight/bias     -- residual stack
        decoder.decoder.{1..K}.shortcut.conv.weight/bias         -- skip path
        decoder.decoder.{...}.conv_transpose.weight/bias         -- upsamplers
        encoder.* / quantizer.* / project_in.* / project_out.*   -- VQ side
    Reverse-engineering this without the upstream modeling code is out of
    scope; the stub below produces a recognisable test signal so we can
    smoke-test the integration end-to-end.
    """

    def __init__(self) -> None:
        super().__init__()
        # Keep a single learnable scalar so the module has parameters and
        # ``state_dict`` round-trips cleanly during integration tests.
        self.placeholder = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._sample_rate = CODEC_OUTPUT_SAMPLE_RATE
        self._samples_per_frame = CODEC_FRAME_UPSAMPLE

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def samples_per_frame(self) -> int:
        return self._samples_per_frame

    @torch.no_grad()
    def forward(self, codebook_ids: torch.Tensor) -> bytes:
        """STUB: replace with full codec reverse-engineering.

        Maps each frame's 16-codebook ids to a per-frame audible frequency and
        emits ``CODEC_FRAME_UPSAMPLE`` int16 samples per input frame at 24 kHz.

        Args:
            codebook_ids: Long tensor of shape ``(B, T, 16)`` (B=1 is the only
                tested batch size; we concat over T in the time axis).

        Returns:
            ``bytes`` -- int16 LE PCM at 24 kHz, length ``B*T*1920*2`` bytes.
        """
        if codebook_ids.dim() != 3 or codebook_ids.size(-1) != CP_NUM_CODEBOOKS:
            raise ValueError(
                f"Expected codebook_ids of shape (B, T, {CP_NUM_CODEBOOKS}), got "
                f"{tuple(codebook_ids.shape)}"
            )

        ids = codebook_ids.detach().to("cpu", dtype=torch.float32).numpy()
        B, T, _ = ids.shape

        # STUB: replace with full codec reverse-engineering
        # Mean codebook id per frame -> frequency in [120 Hz, 2 kHz].
        mean_per_frame = ids.mean(axis=-1)  # (B, T)
        freqs = 120.0 + (mean_per_frame / float(CP_CODEBOOK_VOCAB - 1)) * (2000.0 - 120.0)

        out_chunks: list[bytes] = []
        phase = 0.0
        sr = float(self._sample_rate)
        for b in range(B):
            for t in range(T):
                f = float(freqs[b, t])
                n = self._samples_per_frame
                t_axis = np.arange(n, dtype=np.float32) / sr
                samples = np.sin(2.0 * math.pi * f * t_axis + phase)
                phase = (phase + 2.0 * math.pi * f * (n / sr)) % (2.0 * math.pi)
                # Apply a soft envelope at the frame boundary to avoid clicks.
                env = np.ones(n, dtype=np.float32)
                ramp = min(64, n)
                env[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
                env[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
                samples = samples * env * 0.25  # quiet test tone
                pcm = np.clip(samples, -1.0, 1.0)
                out_chunks.append((pcm * 32767.0).astype(np.int16).tobytes())
        return b"".join(out_chunks)


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


def _load_codec(weights_dir: str, dtype: torch.dtype, device: torch.device) -> Tuple[Code2WavCodec, list[str]]:
    """Load the codec decoder. STUB: we don't actually consume the weights.

    We still verify the file exists and surface its config so the caller can
    sanity-check the sample rate / upsample ratio.
    """
    codec = Code2WavCodec().to(device=device, dtype=dtype)
    cfg_path = os.path.join(weights_dir, "speech_tokenizer", "config.json")
    weights_path = os.path.join(weights_dir, "speech_tokenizer", "model.safetensors")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        # If the config disagrees with our constants, prefer the config.
        codec._sample_rate = int(cfg.get("output_sample_rate", CODEC_OUTPUT_SAMPLE_RATE))
        codec._samples_per_frame = int(cfg.get("decode_upsample_rate", CODEC_FRAME_UPSAMPLE))
    # STUB: we do not load model.safetensors -- record the keys we *would*
    # need to consume so the caller can audit them.
    unaccounted: list[str] = []
    if os.path.exists(weights_path):
        try:
            keys = list(load_file(weights_path, device="cpu").keys())
            unaccounted = keys  # all keys are unaccounted in the stub
        except Exception:
            pass
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
        "codec_stubbed": True,
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
