#!/bin/bash
set -e
echo "=== Setting up cost optimization project on Apple Silicon ==="
pip install --upgrade pip
pip install vllm
pip install httpx pandas matplotlib pyyaml tqdm
pip install pytest pytest-asyncio
python -c "
from huggingface_hub import snapshot_download
print('Downloading model tokenizers...')
snapshot_download('Qwen/Qwen3.5-0.8B')
snapshot_download('Qwen/Qwen3.5-4B')
print('Done. (9B cloud model accessed remotely during benchmark)')
"
echo "=== Setup complete ==="
echo "Start local models:"
echo "  vllm serve Qwen/Qwen3.5-0.8B --port 8010"
echo "  vllm serve Qwen/Qwen3.5-4B --port 8011"
echo "Then run: python -m src.main --profile local"
