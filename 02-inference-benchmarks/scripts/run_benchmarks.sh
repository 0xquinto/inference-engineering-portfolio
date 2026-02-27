#!/bin/bash
# Run the full benchmark suite.
# Run: bash scripts/run_benchmarks.sh [engine]
set -e

ENGINE="${1:-all}"
CONFIG="configs/engines.yaml"
PROMPTS="configs/prompts.json"
OUTPUT="results/"

echo "=== Running Inference Stack Benchmarks ==="
echo "Engine: $ENGINE"
echo "Config: $CONFIG"
echo "Output: $OUTPUT"

nvidia-smi

python -m src.main \
    --engine "$ENGINE" \
    --config "$CONFIG" \
    --prompts "$PROMPTS" \
    --output "$OUTPUT"

echo ""
echo "=== Benchmark complete ==="
echo "Results in: $OUTPUT"
echo "Generate plots: python -c \"from src.visualization import generate_all_plots; generate_all_plots('$OUTPUT/benchmark_*.json')\""
