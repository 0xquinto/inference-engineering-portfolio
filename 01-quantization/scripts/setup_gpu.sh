#!/bin/bash
set -e

echo "=== Setting up quantization project on GPU instance ==="

# Link HF cache if on RunPod
if [ -d "/workspace" ]; then
    mkdir -p /workspace/hf_cache
    ln -sf /workspace/hf_cache /root/.cache/huggingface
    echo "Linked HF cache to /workspace/hf_cache"
fi

# Install Python dependencies
pip install --upgrade pip
pip install llmcompressor>=0.9.0
pip install vllm>=0.16.0
pip install httpx pandas matplotlib pyyaml tqdm
pip install transformers datasets torch
pip install pytest pytest-asyncio

# Download model
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Qwen2.5-7B-Instruct...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype='auto')
print('Done.')
"

echo "=== Setup complete ==="
