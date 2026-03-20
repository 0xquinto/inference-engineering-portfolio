#!/bin/bash
set -e

# =============================================================================
# GPU Benchmark Runner — runs all 6 projects on a single RunPod A40
# Estimated time: ~1.5 hours, cost: ~$1.10 at $0.75/hr
# =============================================================================

echo "============================================"
echo " Inference Engineering Portfolio — GPU Run"
echo "============================================"
echo ""

REPO_DIR="/workspace/inference-engineering-portfolio"
HF_CACHE="/workspace/hf_cache"

# --- Setup ---
echo "=== Phase 0: Environment Setup ==="
mkdir -p $HF_CACHE
ln -sf $HF_CACHE /root/.cache/huggingface

pip install --upgrade pip
pip install vllm>=0.16.0
pip install llmcompressor>=0.9.0
pip install httpx pandas matplotlib pyyaml tqdm
pip install transformers datasets torch
pip install pytest pytest-asyncio

cd $REPO_DIR

# Download models upfront
echo ""
echo "=== Phase 0.5: Download Models ==="
python3 -c "
from huggingface_hub import snapshot_download
for model in ['Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-0.8B']:
    print(f'Downloading {model}...')
    snapshot_download(model)
print('Done.')
"

# Helper: start vLLM and wait for it
start_vllm() {
    local model=$1
    local port=$2
    shift 2
    local extra_args="$@"
    echo "  Starting vLLM: $model on port $port $extra_args"
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$model" --port "$port" \
        --max-model-len 8192 \
        --disable-log-requests \
        $extra_args &
    VLLM_PID=$!
    echo "  Waiting for server (PID=$VLLM_PID)..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            echo "  Server ready."
            return 0
        fi
        sleep 5
    done
    echo "  ERROR: Server failed to start!"
    return 1
}

stop_vllm() {
    if [ -n "$VLLM_PID" ]; then
        echo "  Stopping vLLM (PID=$VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
        sleep 3
    fi
}

# =============================================================================
# Project 01: Quantization
# =============================================================================
echo ""
echo "============================================"
echo " Project 01: Quantization Pipeline"
echo "============================================"

cd $REPO_DIR/01-quantization

# Step 1: Quantize (no vLLM needed)
echo "--- Step 1: Quantize models ---"
python3 -m src.main --profile gpu --step quantize

# Step 2: Evaluate (no vLLM needed, runs locally with transformers)
echo "--- Step 2: Evaluate quality ---"
python3 -m src.main --profile gpu --step evaluate

# Step 3: Benchmark each format via vLLM
echo "--- Step 3: Benchmark each format ---"
for fmt in bf16 w4a16 fp8; do
    model_path="Qwen/Qwen3.5-9B"
    if [ "$fmt" = "w4a16" ]; then
        model_path="quantized_models/Qwen3.5-9B-w4a16"
    elif [ "$fmt" = "fp8" ]; then
        model_path="quantized_models/Qwen3.5-9B-fp8"
    fi
    start_vllm "$model_path" 8010
    python3 -m src.main --profile gpu --step benchmark --format $fmt
    stop_vllm
done

# Step 4: Visualize
echo "--- Step 4: Visualize ---"
python3 -m src.main --profile gpu --step visualize
echo "  Project 01 done."

# =============================================================================
# Project 03: Prefix Caching
# =============================================================================
echo ""
echo "============================================"
echo " Project 03: Prefix Caching"
echo "============================================"

cd $REPO_DIR/03-prefix-caching

# Run with vLLM (prefix caching ON is a flag)
start_vllm "Qwen/Qwen3.5-9B" 8010
python3 -m src.main --profile gpu --engine vllm
stop_vllm
echo "  Project 03 done."

# =============================================================================
# Project 04: Speculative Decoding
# =============================================================================
echo ""
echo "============================================"
echo " Project 04: Speculative Decoding"
echo "============================================"

cd $REPO_DIR/04-speculative-decoding

# Baseline (no speculation)
echo "--- baseline ---"
start_vllm "Qwen/Qwen3.5-9B" 8010
python3 -m src.main --profile gpu --method baseline --step benchmark
stop_vllm

# N-gram (no extra model)
echo "--- ngram ---"
start_vllm "Qwen/Qwen3.5-9B" 8010 \
    --speculative-model "[ngram]" \
    --num-speculative-tokens 5 \
    --ngram-prompt-lookup-max 5 \
    --ngram-prompt-lookup-min 2
python3 -m src.main --profile gpu --method ngram --step benchmark
stop_vllm

# Draft model (Qwen3.5-0.8B)
echo "--- draft_model ---"
start_vllm "Qwen/Qwen3.5-9B" 8010 \
    --speculative-model "Qwen/Qwen3.5-0.8B" \
    --num-speculative-tokens 5
python3 -m src.main --profile gpu --method draft_model --step benchmark
stop_vllm

# MTP (native multi-token prediction)
echo "--- mtp ---"
start_vllm "Qwen/Qwen3.5-9B" 8010 \
    --speculative-model "[mtp]" \
    --num-speculative-tokens 1
python3 -m src.main --profile gpu --method mtp --step benchmark
stop_vllm

# Visualize
python3 -m src.main --profile gpu --step visualize
echo "  Project 04 done."

# =============================================================================
# Project 05: Structured Output
# =============================================================================
echo ""
echo "============================================"
echo " Project 05: Structured Output"
echo "============================================"

cd $REPO_DIR/05-structured-output

start_vllm "Qwen/Qwen3.5-9B" 8010
python3 -m src.main --profile gpu
stop_vllm
echo "  Project 05 done."

# =============================================================================
# Project 06: Cost Optimization
# =============================================================================
echo ""
echo "============================================"
echo " Project 06: Cost Optimization"
echo "============================================"

cd $REPO_DIR/06-cost-optimization

# Start small model
echo "--- Starting 0.8B on port 8010 ---"
start_vllm "Qwen/Qwen3.5-0.8B" 8010
SMALL_PID=$VLLM_PID

# Start medium model
echo "--- Starting 9B on port 8011 ---"
start_vllm "Qwen/Qwen3.5-9B" 8011
MEDIUM_PID=$VLLM_PID

# Skip large (27B) — needs more VRAM than A40 in BF16
# Run with just small + medium tiers
python3 -m src.main --profile gpu --step benchmark --step analyze --step visualize 2>&1 || \
python3 -m src.main --profile gpu

# Stop both
VLLM_PID=$SMALL_PID; stop_vllm
VLLM_PID=$MEDIUM_PID; stop_vllm
echo "  Project 06 done."

# =============================================================================
# Package results
# =============================================================================
echo ""
echo "============================================"
echo " Packaging Results"
echo "============================================"

cd $REPO_DIR
tar czf /workspace/gpu_results.tar.gz \
    01-quantization/results/ \
    03-prefix-caching/results/ \
    04-speculative-decoding/results/ \
    05-structured-output/results/ \
    06-cost-optimization/results/

echo ""
echo "============================================"
echo " ALL DONE"
echo "============================================"
echo "Results packaged: /workspace/gpu_results.tar.gz"
echo "Download with: runpodctl receive /workspace/gpu_results.tar.gz"
