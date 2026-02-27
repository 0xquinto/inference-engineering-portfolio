# 01 — Agentic Inference Gateway

Multi-model serving with intelligent routing optimized for agentic workloads.

## Why This Project

This is your differentiator. Most inference engineers come from ML/systems — they don't
understand agent workloads. You do. This project proves you can serve models AND understand
how they're consumed.

Maps directly to roles at: LiteLLM, Baseten, Together AI, Anyscale.

## What You're Building

```
┌─────────────────────────────────────────────┐
│              Agentic Gateway API             │
│                (FastAPI)                     │
├─────────────┬───────────────┬───────────────┤
│   Router    │  Tool Parser  │   Streaming   │
│ (complexity │  (detect fn   │   (SSE with   │
│  classifier)│   calls)      │   tool chunks)│
├─────────────┴───────────────┴───────────────┤
│           vLLM Serving Layer                │
├──────────────────┬──────────────────────────┤
│  Llama 3.1 8B    │   Llama 3.1 70B AWQ     │
│  (fast/cheap)    │   (complex queries)      │
└──────────────────┴──────────────────────────┘
```

## Key Features to Implement

1. **Smart routing** — Classify request complexity, route to small vs large model
2. **Tool-call aware streaming** — Parse function calls from streaming output
3. **Structured output enforcement** — JSON mode with schema validation
4. **Cost dashboard** — Track tokens, latency, cost savings vs single-model

## Directory Structure

```
01-agentic-gateway/
├── src/
│   ├── router/
│   │   ├── __init__.py
│   │   ├── classifier.py      # Request complexity classification
│   │   └── strategy.py        # Routing logic (cost vs latency vs quality)
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── engine.py           # vLLM engine wrapper
│   │   ├── models.py           # Model configs and loading
│   │   └── streaming.py        # SSE streaming with tool-call detection
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── metrics.py          # Prometheus metrics + cost tracking
│   ├── agents/
│   │   ├── __init__.py
│   │   └── tool_parser.py      # Parse tool calls from model output
│   └── main.py                 # FastAPI app entrypoint
├── configs/
│   ├── models.yaml             # Model configs (paths, quantization, max_tokens)
│   └── routing.yaml            # Routing rules and thresholds
├── benchmarks/
│   └── load_test.py            # Simulate concurrent agent requests
├── scripts/
│   ├── setup_gpu.sh            # GPU instance setup script
│   ├── download_models.sh      # Pull models from HF
│   └── run_server.sh           # Launch the gateway
├── tests/
│   ├── test_router.py
│   ├── test_streaming.py
│   └── test_tool_parser.py
└── requirements.txt
```

## Where to Start

### Day 1: Get a single model serving
```bash
# On your GPU instance
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# Test it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Day 2: Add the FastAPI wrapper + second model
- Build `src/serving/engine.py` — wrap vLLM's AsyncLLMEngine
- Load both 8B and 70B-AWQ models
- Expose a single `/v1/chat/completions` endpoint

### Day 3-4: Build the router
- `src/router/classifier.py` — simple heuristic first (message length, keyword detection)
- Then upgrade to a small classifier (or use the 8B model itself to classify)
- Track routing decisions in metrics

### Day 5-7: Agent-specific features
- Tool-call parsing from streaming output
- Structured output / JSON mode
- Cost tracking dashboard

### Day 8-10: Benchmark and write up
- `benchmarks/load_test.py` — simulate 50 concurrent agent requests
- Compare: gateway (routed) vs always-large-model vs always-small-model
- Show cost savings and latency improvements with real numbers

## Models to Use

| Model | Size | Purpose | Quantization |
|-------|------|---------|-------------|
| Llama 3.1 8B Instruct | 16GB | Fast, simple queries | None (fits A10G) |
| Llama 3.1 70B Instruct | ~36GB | Complex reasoning | AWQ 4-bit |

## Success Metrics (what to show in your writeup)

- Routing accuracy: % of requests correctly classified
- Cost savings: $ saved vs always using the large model
- Latency: P50/P95/P99 for routed vs unrouted
- Throughput: Requests/sec under concurrent load
