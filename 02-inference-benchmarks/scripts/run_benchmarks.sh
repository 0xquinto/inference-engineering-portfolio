#!/bin/bash
# Run the full benchmark suite.
# Uses per-engine venvs to avoid dependency conflicts.
# Run: bash scripts/run_benchmarks.sh [engine]
set -e

ENGINE="${1:-vllm}"
CONFIG="configs/engines.yaml"
PROMPTS="configs/prompts.json"
OUTPUT="results/"

echo "=== Running Inference Stack Benchmarks ==="
echo "Engine: $ENGINE"
echo "Config: $CONFIG"
echo "Output: $OUTPUT"

nvidia-smi

# Select the right Python for each engine
case "$ENGINE" in
    vllm)
        PYTHON="/workspace/venvs/vllm/bin/python"
        ;;
    sglang)
        PYTHON="/workspace/venvs/sglang/bin/python"
        ;;
    tensorrt-llm)
        PYTHON="python"
        ;;
    *)
        echo "Unknown engine: $ENGINE. Use: vllm, sglang, or tensorrt-llm"
        exit 1
        ;;
esac

echo "Using Python: $PYTHON"

$PYTHON -m src.main \
    --engine "$ENGINE" \
    --config "$CONFIG" \
    --prompts "$PROMPTS" \
    --output "$OUTPUT"

echo ""
echo "=== Benchmark complete ==="
echo "Results in: $OUTPUT"
