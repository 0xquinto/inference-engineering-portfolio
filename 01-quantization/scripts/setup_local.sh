#!/bin/bash
set -e
echo "=== Setting up quantization project on Apple Silicon ==="
pip install --upgrade pip
pip install mlx-lm
pip install vllm
pip install httpx pandas matplotlib pyyaml tqdm
pip install pytest pytest-asyncio
python -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen3.5-4B...')
snapshot_download('Qwen/Qwen3.5-4B')
print('Done.')
"
echo "=== Setup complete ==="
echo "Run: python -m src.main --profile local --step quantize"
