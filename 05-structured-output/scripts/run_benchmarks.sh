#!/bin/bash
set -e

echo "=== Running structured output benchmarks ==="
cd "$(dirname "$0")/.."

python3 -m src.main --profile gpu --step benchmark
echo ""

python3 -m src.main --profile gpu --step visualize
echo ""

echo "=== Benchmarks complete. Results in results/ ==="
