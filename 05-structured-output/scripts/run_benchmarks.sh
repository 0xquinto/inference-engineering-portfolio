#!/bin/bash
set -e

echo "=== Running structured output benchmarks ==="
cd "$(dirname "$0")/.."

python -m src.main --step benchmark
echo ""

python -m src.main --step visualize
echo ""

echo "=== Benchmarks complete. Results in results/ ==="
