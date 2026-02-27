#!/bin/bash
# Setup all three inference engines on a RunPod H100 GPU.
# Run: bash scripts/setup_engines.sh
set -e

echo "=== Setting up Inference Benchmark Suite ==="

# Check GPU
nvidia-smi || { echo "No GPU found."; exit 1; }

# Install Python deps
pip install -r requirements.txt

# Login to HF (token should be set)
huggingface-cli whoami || { echo "Not logged in to HF. Run: huggingface-cli login"; exit 1; }

# Download model
echo "=== Downloading Llama 4 Scout ==="
huggingface-cli download meta-llama/Llama-4-Scout-17B-16E-Instruct
echo "Model downloaded."

# Install vLLM (latest)
echo "=== Installing vLLM ==="
pip install vllm --upgrade
echo "vLLM $(python -c 'import vllm; print(vllm.__version__)')"

# Install SGLang
echo "=== Installing SGLang ==="
pip install "sglang[all]" --upgrade
echo "SGLang installed."

# Pull TensorRT-LLM Docker image
echo "=== Pulling TensorRT-LLM ==="
docker pull nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3 || echo "Docker pull failed — TRT-LLM benchmarks may not work."

echo "=== Setup complete ==="
echo "Run: bash scripts/run_benchmarks.sh"
