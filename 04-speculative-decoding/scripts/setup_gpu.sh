#!/bin/bash
set -e

echo "=== Setting up speculative decoding project on GPU instance ==="

# Link HF cache if on RunPod
if [ -d "/workspace" ]; then
    mkdir -p /workspace/hf_cache
    ln -sf /workspace/hf_cache /root/.cache/huggingface
    echo "Linked HF cache to /workspace/hf_cache"
fi

# Install Python dependencies
pip install --upgrade pip
pip install vllm>=0.16.0
pip install httpx pandas matplotlib pyyaml tqdm
pip install pytest pytest-asyncio

# Download model tokenizers
python -c "
from transformers import AutoTokenizer
print('Downloading Qwen2.5-7B-Instruct tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print('Downloading Qwen2.5-0.5B-Instruct (draft model)...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
print('Done.')
"

echo "=== Setup complete ==="
