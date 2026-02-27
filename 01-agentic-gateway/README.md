# Agentic Inference Gateway

Multi-model inference gateway with intelligent routing — **62.5% cost savings** by routing simple requests to a small model and complex ones to a large model on a single GPU.

## Architecture

```
                        ┌───────────────────────────────────────────┐
  Client Request ──────▶│          Agentic Gateway (FastAPI)        │
  (OpenAI-compat)       ├────────────┬──────────────┬──────────────┤
                        │  5-Signal  │  Tool Call   │  SSE         │
                        │  Classifier│  Parser      │  Streaming   │
                        ├────────────┴──────────────┴──────────────┤
                        │           vLLM Serving Layer              │
                        ├───────────────────┬───────────────────────┤
                        │  Qwen 2.5 7B      │  Qwen 2.5 32B AWQ   │
                        │  (40% VRAM)       │  (50% VRAM)          │
                        └───────────────────┴───────────────────────┘
                                    RunPod A40 (48 GB)
```

## Key Results

| Metric | Value |
|--------|-------|
| End-to-end success rate | 60/60 requests |
| Cost savings vs all-large | 62.5% |
| P50 latency | 1,813 ms |
| P95 latency | 10,596 ms |
| Routing split | 75% small, 25% large |
| GPU | Single NVIDIA A40 48 GB |

## How It Works

The gateway scores each incoming request on a 0–1 complexity scale using five weighted signals: **message length** (0.2), **keyword complexity** (0.35), **tool use** (0.2), **conversation depth** (0.1), and **structured output** (0.15). Requests scoring below 0.6 route to the fast 7B model; those above route to the 32B model. Explicit model overrides and automatic fallback (when one model is unavailable) take priority over the classifier.

## Routing Breakdown

- **Simple queries** → Qwen 7B (greeting, short factual questions, single-turn chat)
- **Complex reasoning** → Qwen 32B AWQ (multi-step analysis, code generation, detailed explanations)
- **Tool-call requests** → Qwen 7B with native tool support (function calling via Hermes format)

## Project Structure

```
01-agentic-gateway/
├── src/
│   ├── main.py                    # FastAPI entrypoint, /v1/chat/completions endpoint
│   ├── router/
│   │   ├── classifier.py          # 5-signal complexity scorer (0.0–1.0)
│   │   └── strategy.py            # Override → classify → fallback routing logic
│   ├── serving/
│   │   ├── engine.py              # httpx wrapper around vLLM servers
│   │   ├── models.py              # YAML-driven model + cost registry
│   │   └── streaming.py           # SSE streaming with tool-call detection
│   ├── dashboard/
│   │   └── metrics.py             # In-memory cost tracking and latency percentiles
│   └── agents/
│       └── tool_parser.py         # Extracts tool calls from native + JSON-in-text formats
├── configs/
│   ├── models.yaml                # Model names, VRAM splits, cost rates
│   └── routing.yaml               # Classifier weights and thresholds
├── tests/
│   ├── conftest.py                # Shared fixtures
│   ├── test_router.py             # Classifier scoring and edge cases (9 tests)
│   ├── test_strategy.py           # Override, fallback, complexity routing (7 tests)
│   ├── test_tool_parser.py        # Native, JSON fallback, streaming deltas (7 tests)
│   ├── test_metrics.py            # Cost savings, percentiles, routing splits (6 tests)
│   ├── test_models.py             # CostConfig math, YAML loading (5 tests)
│   └── test_engine.py             # Health checks, generate, retry logic (7 tests)
├── benchmarks/
│   └── load_test.py               # Concurrent agent request simulator
├── scripts/
│   ├── setup_gpu.sh               # Install deps + download Qwen models
│   └── run_server.sh              # Launch vLLM servers + gateway
└── requirements.txt
```

## Deployment

Deployed on **RunPod** with a single NVIDIA A40 (48 GB VRAM):

- Qwen 2.5 7B Instruct — port 8001, 40% GPU memory
- Qwen 2.5 32B Instruct AWQ (INT4) — port 8002, 50% GPU memory
- Gateway — port 8080

```bash
# On GPU instance
bash scripts/setup_gpu.sh          # install deps + download models
MODE=multi bash scripts/run_server.sh   # start both models + gateway
```

## Running Tests

All 41 tests run locally without a GPU (engine tests use mocked httpx):

```bash
pytest tests/ -v
```

## Design Decisions

**Why Qwen 2.5?** Apache 2.0 licensed, ungated on Hugging Face (no access request delays), strong tool-calling support via Hermes format, and the 32B AWQ variant fits alongside the 7B on a single 48 GB GPU.

**Why separate vLLM processes?** Each model runs as its own vLLM server with isolated GPU memory allocation. This avoids contention, allows independent restarts, and mirrors how multi-model serving works in production.

**Why a heuristic classifier?** A simple weighted-signal approach is fast (< 1 ms), interpretable, and easy to tune without training data. It avoids adding model inference overhead to the routing decision. The five signals cover the most common complexity indicators for agentic workloads.
