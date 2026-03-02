#!/bin/bash
set -e

echo "=== Running full quantization pipeline ==="

cd "$(dirname "$0")/.."

# Step 1: Quantize
python -m src.main --step quantize
echo ""

# Step 2: Evaluate quality (runs each model on GPU)
python -m src.main --step evaluate
echo ""

# Step 3: Benchmark each format via vLLM (starts/stops server per format)
# Note: this step requires manually starting vLLM with each model.
# See README for per-format benchmark instructions.
echo "Step 3 (benchmark) requires starting vLLM per format."
echo "Run: python -m src.main --step benchmark --format <format_name>"
echo ""

# Step 4: Visualize
python -m src.main --step visualize
echo ""

echo "=== Pipeline complete. Results in results/ ==="
