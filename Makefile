# Makefile for e3-megakernel-tts
# Tab-indented (Make requirement). All targets are .PHONY -- there are no
# real build artifacts tracked by Make.

.DEFAULT_GOAL := help

.PHONY: help install weights build-kernel bench ui demo demo-stub smoke-test clean

help: ## Print all targets with one-line descriptions
	@echo "e3-megakernel-tts -- Make targets"
	@echo ""
	@echo "  make install       Install Python dependencies (inference-server/requirements.txt + safetensors)"
	@echo "  make weights       Download Qwen3-TTS-1.7B-CustomVoice weights to /workspace/qwen3-tts-1.7b"
	@echo "  make build-kernel  JIT-compile the modified qwen_megakernel CUDA extension"
	@echo "  make bench         Run the talker decode-loop bench (TTFC, RTF, tok/s)"
	@echo "  make ui            Launch the Gradio UI v2 dashboard"
	@echo "  make demo          Run the Pipecat voice loop (mic -> STT -> LLM -> our TTS -> speaker)"
	@echo "  make demo-stub     Run pipecat_demo.py with stubbed megakernel + file I/O (offline smoke)"
	@echo "  make smoke-test    py_compile every Python file (no GPU required)"
	@echo "  make clean         Remove __pycache__, *.pyc, samples/bot_response.wav"

install: ## Install Python dependencies
	pip install -r inference-server/requirements.txt safetensors

weights: ## Download Qwen3-TTS-1.7B-CustomVoice weights
	huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir /workspace/qwen3-tts-1.7b

build-kernel: ## JIT-compile qwen_megakernel from the modified fork
	cd qwen_megakernel_modified && python3 -c "import qwen_megakernel"

bench: ## Run the talker decode-loop benchmark with default flags
	python3 inference-server/bench_megakernel.py

ui: ## Launch the Gradio UI v2
	python3 inference-server/ui_v2.py

demo: ## Run the full Pipecat voice loop
	python3 inference-server/pipecat_demo.py

demo-stub: ## Run pipecat_demo.py with stubbed megakernel + file I/O
	MEGAKERNEL_STUB=1 INPUT_MODE=file INPUT_WAV=samples/user_utterance.wav OUTPUT_WAV=samples/bot_response.wav python3 inference-server/pipecat_demo.py

smoke-test: ## py_compile sanity check on all Python sources
	python3 -m py_compile inference-server/*.py qwen_megakernel_modified/qwen_megakernel/*.py

clean: ## Remove __pycache__, *.pyc, generated sample outputs
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -f samples/bot_response.wav
