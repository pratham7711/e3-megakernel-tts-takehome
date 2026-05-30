# NOTICES

Third-party license attribution for code included or referenced by this repo.
Full license texts live in the upstream `LICENSE` files at the paths linked
below; this document only summarises the SPDX identifier, copyright line, and
what we use.

---

## AlpinDale -- `qwen_megakernel` (original baseline)

- **SPDX**: `MIT`
- **Copyright**: `Copyright (c) 2026 AlpinDale`
- **Location in this repo**: [`qwen_megakernel/`](./qwen_megakernel/) (clean
  clone, read-only reference)
- **License file**: [`qwen_megakernel/LICENSE`](./qwen_megakernel/LICENSE)
- **Upstream**: <https://github.com/AlpinDale/qwen_megakernel>
- **What we use**: The CUDA megakernel architecture, the C++/PyTorch
  bindings, the Python build glue, and the bench harness skeleton. Shipped
  unmodified as a reference clone alongside our fork.

## AlpinDale derivative -- `qwen_megakernel_modified` (our fork)

- **SPDX**: `MIT` (preserves upstream license per MIT requirements)
- **Copyright**: `Copyright (c) 2026 AlpinDale` (upstream) +
  modifications by Pratham Sharma, 2026
- **Location in this repo**:
  [`qwen_megakernel_modified/`](./qwen_megakernel_modified/)
- **License file**:
  [`qwen_megakernel_modified/LICENSE`](./qwen_megakernel_modified/LICENSE)
- **What we modified**: model shape constants in `csrc/kernel.cu`, the Python
  weight loader in `qwen_megakernel/model.py` for Qwen3-TTS-1.7B talker
  weights, MRoPE cos/sin table builder, and `prefill_text()` for pure-Python
  KV-cache prefill. See `ARCHITECTURE.md` Section 3 and `CHANGELOG.md` for
  the full diff summary.

## QwenLM -- `Qwen3-TTS` (codec architecture reference)

- **SPDX**: `Apache-2.0`
- **Copyright**: `Copyright 2025-2026 The Qwen Team, Alibaba Group`
- **Location in this repo**: not vendored; only the model weights are
  downloaded at runtime via `huggingface-cli download
  Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`.
- **Upstream LICENSE**:
  <https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE>
- **What we use**: The Qwen3-TTS-1.7B-CustomVoice model weights at inference
  time, and the codec architecture **as a reference for our clean-room
  reimplementation** in
  [`inference-server/qwen3_tts_components.py`](./inference-server/qwen3_tts_components.py).
  We did not vendor the `qwen-tts` pip package (torchaudio ABI conflicts on
  the PyTorch nightly NGC image); the file is informed by upstream modelling
  code but written against `torch` only.

## Daily / Pipecat -- `pipecat` (real-time voice framework, reference clone)

- **SPDX**: `BSD-2-Clause`
- **Copyright**: `Copyright (c) 2024-2026, Daily`
- **Location in this repo**: [`pipecat/`](./pipecat/) (reference clone, NOT
  shipped as part of the inference path; an `inference-server/requirements.txt`
  pip install pulls the published `pipecat-ai` package independently)
- **License file**: [`pipecat/LICENSE`](./pipecat/LICENSE)
- **Upstream**: <https://github.com/pipecat-ai/pipecat>
- **What we use**: The `TTSService` base class, `FrameProcessor` lifecycle,
  `Pipeline` + `PipelineTask` orchestration, `SileroVADAnalyzer`,
  `LocalAudioOutputTransport`, `DeepgramSTTService`, `GroqLLMService`, and
  the `LLMContextAggregatorPair` user-turn pattern. All imported from the
  installed `pipecat-ai` PyPI package at runtime; the local clone is for
  source reference only.

---

*This file lists upstream dependencies whose code or architecture we
incorporate; the standard transitive pip dependencies pulled in by
`inference-server/requirements.txt` (torch, transformers, numpy,
soundfile, gradio, loguru, python-dotenv, huggingface-hub) carry their own
licenses and are not enumerated here. See each package's PyPI page for
upstream license terms.*
