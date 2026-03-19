#!/bin/bash
set -e

echo "=== Setting up cost optimization project on GPU instance ==="

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
print('Downloading model tokenizers...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print('Done. (72B model downloaded on-demand during benchmark)')
"

echo "=== Setup complete ==="
