#!/bin/bash
set -e

# =============================================================================
# GPU Speculative Decoding — runs all methods on Qwen3.5-9B
# Estimated time: ~30 min, cost: ~$0.40
# =============================================================================

REPO_DIR="/workspace/inference-engineering-portfolio"
cd $REPO_DIR

pip install -q httpx pyyaml pandas matplotlib tqdm

# Helper: start vLLM and wait
start_vllm() {
    local model=$1
    local port=$2
    shift 2
    local extra_args="$@"
    echo "  Starting vLLM: $model on port $port $extra_args"
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$model" --port "$port" \
        --max-model-len 8192 \
        $extra_args > /workspace/vllm_server.log 2>&1 &
    VLLM_PID=$!
    for i in $(seq 1 120); do
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            echo "  Server ready (PID=$VLLM_PID)."
            return 0
        fi
        sleep 5
    done
    echo "  ERROR: Server failed to start!"
    cat /workspace/vllm_server.log | tail -20
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

cd $REPO_DIR/04-speculative-decoding

# --- Baseline (already have this, but rerun for consistency) ---
echo "=== baseline ==="
start_vllm "Qwen/Qwen3.5-9B" 8010
python3 -m src.main --profile gpu --method baseline --step benchmark
stop_vllm

# --- N-gram (no extra model needed) ---
echo "=== ngram ==="
start_vllm "Qwen/Qwen3.5-9B" 8010 \
    --speculative-model "[ngram]" \
    --num-speculative-tokens 5 \
    --ngram-prompt-lookup-max 5 \
    --ngram-prompt-lookup-min 2
python3 -m src.main --profile gpu --method ngram --step benchmark
stop_vllm

# --- Draft model (Qwen3.5-0.8B) ---
echo "=== draft_model ==="
start_vllm "Qwen/Qwen3.5-9B" 8010 \
    --speculative-model "Qwen/Qwen3.5-0.8B" \
    --num-speculative-tokens 5
python3 -m src.main --profile gpu --method draft_model --step benchmark
stop_vllm

# --- MTP (native multi-token prediction) ---
echo "=== mtp ==="
start_vllm "Qwen/Qwen3.5-9B" 8010 \
    --speculative-model "[mtp]" \
    --num-speculative-tokens 1
python3 -m src.main --profile gpu --method mtp --step benchmark
stop_vllm

# --- Visualize ---
python3 -m src.main --profile gpu --step visualize

echo ""
echo "=== ALL SPEC DECODING METHODS DONE ==="
echo "Results: $REPO_DIR/04-speculative-decoding/results/"
cat results/speculative_results.json
