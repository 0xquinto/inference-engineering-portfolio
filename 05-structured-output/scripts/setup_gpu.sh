#!/bin/bash
set -e

echo "=== Setting up structured output project on GPU instance ==="

if [ -d "/workspace" ]; then
    mkdir -p /workspace/hf_cache
    ln -sf /workspace/hf_cache /root/.cache/huggingface
    echo "Linked HF cache to /workspace/hf_cache"
fi

pip install --upgrade pip
pip install vllm>=0.16.0
pip install httpx pandas matplotlib pyyaml tqdm
pip install pytest pytest-asyncio

python -c "
from transformers import AutoTokenizer
print('Downloading Qwen2.5-7B-Instruct...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print('Done.')
"

echo "=== Setup complete ==="
