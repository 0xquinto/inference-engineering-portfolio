#!/bin/bash
# Launch vLLM model servers + the gateway.
# Run on your GPU instance after setup_gpu.sh.
set -e

echo "=== Starting Agentic Inference Gateway ==="

# Check GPU
nvidia-smi > /dev/null 2>&1 || { echo "No GPU found."; exit 1; }

# --- Option A: Single GPU (A10G 24GB) ---
# Only run the small model. Good for development.
if [ "${MODE:-single}" = "single" ]; then
    echo "[1/2] Starting small model (Qwen 7B) on port 8001..."
    vllm serve Qwen/Qwen2.5-7B-Instruct \
        --port 8001 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        &
    SMALL_PID=$!
    sleep 30  # wait for model to load

    echo "[2/2] Starting gateway on port 8080..."
    SMALL_MODEL_URL=http://localhost:8001 \
    LARGE_MODEL_URL=http://localhost:9999 \
    python -m uvicorn src.main:app --host 0.0.0.0 --port 8080
fi

# --- Option B: Multi-GPU (A100 80GB or 2x A10G) ---
# Run both small and large models.
if [ "$MODE" = "multi" ]; then
    echo "[1/3] Starting small model (Qwen 7B) on port 8001..."
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
        --port 8001 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.40 \
        --enable-auto-tool-choice --tool-call-parser hermes \
        &
    SMALL_PID=$!

    echo "[2/3] Starting large model (Qwen 32B AWQ) on port 8002..."
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
        --port 8002 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.50 \
        --quantization awq \
        --enable-auto-tool-choice --tool-call-parser hermes \
        &
    LARGE_PID=$!

    echo "Waiting for models to load..."
    sleep 60

    echo "[3/3] Starting gateway on port 8080..."
    SMALL_MODEL_URL=http://localhost:8001 \
    LARGE_MODEL_URL=http://localhost:8002 \
    python -m uvicorn src.main:app --host 0.0.0.0 --port 8080
fi

# Cleanup on exit
trap "kill $SMALL_PID $LARGE_PID 2>/dev/null" EXIT
