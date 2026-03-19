#!/bin/bash
set -e

echo "=== Setting up speculative decoding project on Apple Silicon ==="

pip install --upgrade pip
pip install vllm
pip install httpx pandas matplotlib pyyaml tqdm
pip install pytest pytest-asyncio

python -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen3.5-4B...')
snapshot_download('Qwen/Qwen3.5-4B')
print('Downloading Qwen3.5-0.8B (draft model)...')
snapshot_download('Qwen/Qwen3.5-0.8B')
print('Done.')
"

echo "=== Setup complete. Start server with: ==="
echo "vllm serve Qwen/Qwen3.5-4B --port 8010"
echo "Then run: python -m src.main --profile local"
