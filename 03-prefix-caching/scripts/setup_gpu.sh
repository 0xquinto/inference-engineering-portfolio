#!/bin/bash
set -e

echo "=== Setting up prefix caching project on GPU instance ==="

if [ -d "/workspace" ]; then
    mkdir -p /workspace/hf_cache
    ln -sf /workspace/hf_cache /root/.cache/huggingface
fi

pip install --upgrade pip
pip install vllm>=0.16.0
pip install httpx pandas matplotlib pyyaml tqdm
pip install pytest pytest-asyncio

echo "Downloading model..."
python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B')
"

echo "=== Setup complete ==="
