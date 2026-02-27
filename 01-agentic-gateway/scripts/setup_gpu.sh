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
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"
echo "Qwen 7B downloaded."

python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-32B-Instruct-AWQ')"
echo "Qwen 32B AWQ downloaded."

echo "=== Setup complete ==="
echo "Start with: MODE=multi bash scripts/run_server.sh"
