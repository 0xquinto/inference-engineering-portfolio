#!/bin/bash
# Run this on your GPU instance (RunPod, Lambda, etc.)
set -e

echo "=== Setting up Agentic Gateway ==="

# Check GPU
nvidia-smi || { echo "No GPU found. This project requires a GPU instance."; exit 1; }

# Install dependencies
pip install -r requirements.txt

# Download models (this takes a while)
echo "=== Downloading models ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B-Instruct')"
echo "8B model downloaded."

# For 70B AWQ, you need at least 40GB VRAM (A100) or use a smaller quantized version
# python -c "from huggingface_hub import snapshot_download; snapshot_download('hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4')"

echo "=== Setup complete ==="
echo "Start with: python src/main.py"
