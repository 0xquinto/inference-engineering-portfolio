# Inference Stack Benchmarks

Rigorous, reproducible comparison of **vLLM**, **SGLang**, and **TensorRT-LLM** serving Llama 4 Scout 17B-16E (MoE) on a single NVIDIA H100 80GB GPU.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         Benchmark Suite          │
                    │  async httpx · streaming SSE     │
                    │  concurrency: 1, 10, 50, 100    │
                    └──────────┬──────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
     ┌──────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐
     │    vLLM     │   │   SGLang    │   │ TensorRT-LLM│
     │   v0.16+    │   │   v0.5+     │   │  v1.3+ (Docker)
     │  port 8001  │   │  port 8002  │   │  port 8003  │
     └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │   NVIDIA H100 SXM   │
                    │      80GB HBM3      │
                    └─────────────────────┘
```

## Model

**Llama 4 Scout 17B-16E-Instruct** — Meta's Mixture-of-Experts model with 16 experts, ~109B total parameters but only ~17B active per forward pass. Chosen because MoE models are the dominant architecture for production inference and stress-test engine scheduling in ways dense models don't.

## Metrics

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to first token (ms) — measures prefill latency |
| **TPOT** | Time per output token (ms) — measures decode speed |
| **Throughput** | Tokens/sec at each concurrency level |
| **P50 / P95 / P99** | Latency percentiles under load |
| **GPU Memory** | VRAM usage (MB) via nvidia-smi |

## How It Works

1. **Runner interface** — Each engine implements `BenchmarkRunner` (start server, health check, send request, stop server)
2. **Streaming metrics** — Requests use SSE streaming to measure TTFT separately from total latency
3. **Concurrent load** — `asyncio.Semaphore` controls concurrency; `asyncio.gather` fires requests in parallel
4. **Prompt categories** — Short, medium, long, and code prompts test different workload profiles
5. **Results** — Raw JSON with per-request metrics, aggregated into latency/throughput summaries and matplotlib charts

## Project Structure

```
02-inference-benchmarks/
├── src/
│   ├── runners/
│   │   ├── base.py              # Abstract BenchmarkRunner + RequestConfig + BenchmarkResult
│   │   ├── vllm_runner.py       # vLLM subprocess runner
│   │   ├── sglang_runner.py     # SGLang subprocess runner
│   │   └── trtllm_runner.py     # TensorRT-LLM Docker runner
│   ├── metrics/
│   │   ├── latency.py           # LatencyTracker — TTFT, TPOT, percentiles
│   │   ├── throughput.py        # ThroughputCalculator — tokens/sec
│   │   └── memory.py            # nvidia-smi GPU memory parsing
│   ├── visualization/
│   │   └── plots.py             # Latency bar charts, throughput scaling curves
│   └── main.py                  # CLI entrypoint with engine orchestration
├── configs/
│   ├── engines.yaml             # Engine ports, model config, benchmark params
│   └── prompts.json             # 16 prompts across 4 categories
├── scripts/
│   ├── setup_engines.sh         # Install engines + download model on GPU box
│   └── run_benchmarks.sh        # Run full benchmark suite
├── tests/                       # 16 unit tests (metrics, runners, CLI, visualization)
├── results/                     # Benchmark JSON + charts (generated)
└── requirements.txt
```

## Quick Start

### On a GPU machine (RunPod H100 recommended)

```bash
# 1. Clone and enter project
git clone https://github.com/0xquinto/inference-engineering-portfolio.git
cd inference-engineering-portfolio/02-inference-benchmarks

# 2. Setup engines and download model
bash scripts/setup_engines.sh

# 3. Run benchmarks (one engine at a time to avoid OOM)
bash scripts/run_benchmarks.sh vllm
bash scripts/run_benchmarks.sh sglang
bash scripts/run_benchmarks.sh tensorrt-llm

# 4. Generate comparison charts
python -c "from src.visualization import generate_all_plots; import glob; generate_all_plots(glob.glob('results/benchmark_*.json')[0])"
```

### Local development (no GPU needed)

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python -m src.main --list-engines
```

## Running Tests

```bash
python -m pytest tests/ -v
```

```
tests/test_main.py          — 2 tests (CLI help, engine listing)
tests/test_metrics.py       — 6 tests (latency percentiles, throughput, GPU memory parsing)
tests/test_runner_init.py   — 3 tests (all runners instantiate correctly)
tests/test_runners.py       — 3 tests (dataclass fields, config defaults, ABC enforcement)
tests/test_plots.py         — 2 tests (data preparation for charts)
────────────────────────────────────────────────────────
Total: 16 passed
```

## Design Decisions

**Why these 3 engines?** vLLM (PagedAttention, most popular), SGLang (RadixAttention, fastest for structured output), TensorRT-LLM (NVIDIA's optimized runtime). These are the three stacks teams actually evaluate when choosing inference infrastructure.

**Why Llama 4 Scout (MoE)?** MoE models are becoming the default for production (GPT-4, Mixtral, DBRX). They stress-test engine scheduling, expert routing, and memory management differently than dense models — revealing performance gaps that matter in production.

**Why H100?** It's the standard GPU for production inference. Benchmarking on consumer hardware would produce misleading results that don't transfer to real deployments.

**Why streaming?** TTFT matters more than total latency for user-facing applications. Streaming SSE lets us measure prefill and decode phases independently, which is how production systems are evaluated.

**Why async httpx with semaphore?** Simulates realistic concurrent load without the overhead of spawning threads. The semaphore pattern matches how load balancers dispatch requests to inference servers.
