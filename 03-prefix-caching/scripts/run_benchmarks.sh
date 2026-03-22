#!/bin/bash
set -e

cd "$(dirname "$0")/.."

ENGINE="${1:-vllm}"
MODEL="Qwen/Qwen3.5-9B"
PORT=8010

echo "=== Running prefix caching benchmarks with $ENGINE ==="

start_server() {
    local extra_args="$1"
    echo "  Starting $ENGINE: $MODEL on port $PORT $extra_args"
    if [ "$ENGINE" = "vllm" ]; then
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" --port "$PORT" \
            --host 0.0.0.0 \
            --max-model-len 8192 \
            --no-enable-log-requests \
            $extra_args > /workspace/vllm_server.log 2>&1 &
    elif [ "$ENGINE" = "sglang" ]; then
        python3 -m sglang.launch_server \
            --model-path "$MODEL" --port "$PORT" \
            --context-length 8192 \
            $extra_args > /workspace/sglang_server.log 2>&1 &
    fi
    SERVER_PID=$!

    for i in $(seq 1 120); do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "  Server ready (PID=$SERVER_PID)."
            return 0
        fi
        sleep 5
    done
    echo "  ERROR: Server failed to start!"
    return 1
}

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        echo "  Stopping server..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        sleep 3
    fi
}

# Run WITHOUT prefix caching
echo ""
echo "=== Phase 1: NO prefix caching ==="
if [ "$ENGINE" = "sglang" ]; then
    start_server "--disable-radix-cache"
else
    start_server ""
fi
python3 -m src.main --profile gpu --engine "$ENGINE" 2>&1 | tee results/run_no_cache.log
stop_server

# Run WITH prefix caching
echo ""
echo "=== Phase 2: WITH prefix caching ==="
if [ "$ENGINE" = "sglang" ]; then
    start_server ""
else
    start_server "--enable-prefix-caching"
fi
python3 -m src.main --profile gpu --engine "$ENGINE" 2>&1 | tee results/run_with_cache.log
stop_server

echo ""
echo "=== Benchmarks complete ==="
