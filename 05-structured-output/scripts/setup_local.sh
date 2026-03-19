#!/bin/bash
set -e
echo "=== Setting up structured output project on Apple Silicon ==="
pip install --upgrade pip
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
echo "Start vLLM: vllm serve Qwen/Qwen3.5-4B --port 8010"
echo "Then run: python -m src.main --profile local"
