#!/bin/bash
# Setup all three inference engines on a RunPod H100 GPU.
# vLLM and SGLang have conflicting deps (flashinfer), so each gets its own venv.
# Run: bash scripts/setup_engines.sh
set -e

echo "=== Setting up Inference Benchmark Suite ==="

# Check GPU
nvidia-smi || { echo "No GPU found."; exit 1; }

# Install shared deps (no engine-specific packages)
pip install httpx pandas matplotlib plotly pyyaml tqdm pytest pytest-asyncio nvidia-ml-py huggingface_hub

# Verify HF login
python -c "from huggingface_hub import whoami; print('Logged in as:', whoami()['name'])" || {
    echo "Not logged in to HF. Run: python -c \"from huggingface_hub import login; login(token='YOUR_TOKEN')\""
    exit 1
}

# Download model
echo "=== Downloading Llama 4 Scout ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-4-Scout-17B-16E-Instruct')"
echo "Model downloaded."

# Create vLLM venv
echo "=== Setting up vLLM venv ==="
python -m venv /workspace/venvs/vllm --system-site-packages
/workspace/venvs/vllm/bin/pip install vllm httpx pyyaml
echo "vLLM $(/workspace/venvs/vllm/bin/python -c 'import vllm; print(vllm.__version__)')"

# Create SGLang venv
echo "=== Setting up SGLang venv ==="
python -m venv /workspace/venvs/sglang --system-site-packages
/workspace/venvs/sglang/bin/pip install "sglang[all]" httpx pyyaml
echo "SGLang installed."

# Pull TensorRT-LLM Docker image
echo "=== Pulling TensorRT-LLM ==="
docker pull nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3 || echo "Docker pull failed — TRT-LLM benchmarks may not work."

echo ""
echo "=== Setup complete ==="
echo "Engines installed in separate venvs to avoid dependency conflicts:"
echo "  vLLM:   /workspace/venvs/vllm/bin/python"
echo "  SGLang: /workspace/venvs/sglang/bin/python"
echo ""
echo "Run: bash scripts/run_benchmarks.sh vllm"
