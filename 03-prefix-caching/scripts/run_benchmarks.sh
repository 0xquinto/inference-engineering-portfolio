#!/bin/bash
set -e

cd "$(dirname "$0")/.."

ENGINE="${1:-vllm}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8010

echo "=== Running prefix caching benchmarks with $ENGINE ==="

# Run WITHOUT prefix caching
echo "Starting $ENGINE WITHOUT prefix caching..."
if [ "$ENGINE" = "vllm" ]; then
    vllm serve "$MODEL" --port "$PORT" --max-model-len 8192 &
elif [ "$ENGINE" = "sglang" ]; then
    python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --context-length 8192 --disable-radix-cache &
fi
SERVER_PID=$!

echo "Waiting for server..."
sleep 30
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    sleep 5
done

python -m src.main --engine "$ENGINE" 2>&1 | tee results/run_no_cache.log

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 5

# Run WITH prefix caching
echo "Starting $ENGINE WITH prefix caching..."
if [ "$ENGINE" = "vllm" ]; then
    vllm serve "$MODEL" --port "$PORT" --max-model-len 8192 --enable-prefix-caching &
elif [ "$ENGINE" = "sglang" ]; then
    python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --context-length 8192 &
fi
SERVER_PID=$!

echo "Waiting for server..."
sleep 30
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    sleep 5
done

python -m src.main --engine "$ENGINE" 2>&1 | tee results/run_with_cache.log

kill $SERVER_PID 2>/dev/null || true

echo "=== Benchmarks complete ==="
