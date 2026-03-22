#!/bin/bash
set -e

# =============================================================================
# GPU benchmarks for structured output (P05) and prefix caching (P03)
# Runs both projects in a single RunPod session.
# Estimated time: ~35 min, cost: ~$0.50
#
# Prerequisites:
#   - runpod-torch-v280 template (CUDA 12.8 driver)
#   - pip install --break-system-packages vllm --extra-index-url https://download.pytorch.org/whl/cu128
#   - Qwen3.5-9B downloaded to HF cache
# =============================================================================

REPO_DIR="/workspace/inference-engineering-portfolio"
cd $REPO_DIR

# Install benchmark deps
pip install --break-system-packages -q httpx pyyaml pandas matplotlib tqdm

# Ensure HF cache is linked
mkdir -p /workspace/hf_cache
ln -sf /workspace/hf_cache /root/.cache/huggingface

# Helper: start vLLM and wait for health
# Uses "$@" directly to preserve JSON quoting
start_vllm() {
    local model=$1
    local port=$2
    shift 2
    echo "  Starting vLLM: $model on port $port $@"
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$model" --port "$port" \
        --host 0.0.0.0 \
        --max-model-len 8192 \
        --no-enable-log-requests \
        "$@" > /workspace/vllm_server.log 2>&1 &
    VLLM_PID=$!
    for i in $(seq 1 120); do
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            echo "  Server ready (PID=$VLLM_PID)."
            return 0
        fi
        sleep 5
    done
    echo "  ERROR: Server failed to start!"
    tail -20 /workspace/vllm_server.log
    return 1
}

stop_vllm() {
    if [ -n "$VLLM_PID" ]; then
        echo "  Stopping vLLM..."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
        sleep 3
    fi
}

MODEL="Qwen/Qwen3.5-9B"

# =============================================
# PROJECT 05: Structured Output (~15 min)
# =============================================
echo ""
echo "######################################"
echo "# P05: Structured Output Benchmarks  #"
echo "######################################"
echo ""

cd $REPO_DIR/05-structured-output

# Single server — structured output backends are selected per-request via guided_json
# Disable thinking mode server-side to prevent <think> tags corrupting guided JSON
start_vllm "$MODEL" 8010 --default-chat-template-kwargs '{"enable_thinking": false}'
python3 -m src.main --profile gpu --step benchmark
python3 -m src.main --profile gpu --step visualize
stop_vllm

echo ""
echo "P05 results:"
cat results/structured_output_results.json
echo ""

# =============================================
# PROJECT 03: Prefix Caching (~20 min)
# =============================================
echo ""
echo "######################################"
echo "# P03: Prefix Caching Benchmarks     #"
echo "######################################"
echo ""

cd $REPO_DIR/03-prefix-caching

# Phase 1: WITHOUT prefix caching
echo "=== Phase 1: NO prefix caching ==="
start_vllm "$MODEL" 8010
python3 -m src.main --profile gpu --engine vllm
stop_vllm

# Phase 2: WITH prefix caching
echo "=== Phase 2: WITH prefix caching ==="
start_vllm "$MODEL" 8010 --enable-prefix-caching
python3 -m src.main --profile gpu --engine vllm
stop_vllm

echo ""
echo "P03 results:"
cat results/prefix_caching_results.json
echo ""

echo ""
echo "=== ALL BENCHMARKS DONE ==="
echo "P05 results: $REPO_DIR/05-structured-output/results/"
echo "P03 results: $REPO_DIR/03-prefix-caching/results/"
