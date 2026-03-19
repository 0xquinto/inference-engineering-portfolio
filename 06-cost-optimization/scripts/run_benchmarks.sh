#!/bin/bash
set -e

echo "=== Running cost optimization analysis ==="
cd "$(dirname "$0")/.."

python -m src.main --step benchmark
echo ""

python -m src.main --step analyze
echo ""

python -m src.main --step visualize
echo ""

echo "=== Analysis complete. Results in results/ ==="
