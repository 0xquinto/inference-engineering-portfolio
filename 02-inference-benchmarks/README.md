# 02 — Inference Stack Benchmarks

Rigorous, reproducible comparison of vLLM, SGLang, and TGI serving the same model.

## Why This Project

Companies need engineers who can evaluate and choose inference stacks — not just use them.
This project shows you think in tradeoffs and measure before deciding.

Maps directly to roles at: NVIDIA, AMD (SGLang team), any infra team choosing a stack.

## What You're Measuring

```
Same model (Llama 3.1 8B) across 3 engines:

┌──────────┐  ┌──────────┐  ┌──────────┐
│   vLLM   │  │  SGLang  │  │   TGI    │
│  v0.11+  │  │  v0.4+   │  │  v2.x    │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     └─────────────┼─────────────┘
                   │
          ┌────────┴────────┐
          │  Benchmark Suite │
          │  (same prompts,  │
          │   same hardware) │
          └─────────────────┘
```

## Metrics to Capture

| Metric | What it means |
|--------|---------------|
| TTFT (Time to First Token) | How fast the model starts responding |
| Token throughput (tok/s) | Generation speed |
| Throughput under load | Requests/sec at 1, 10, 50, 100 concurrent |
| Memory usage | VRAM consumed at idle and under load |
| Quantization impact | Same metrics with AWQ vs FP16 |

## Directory Structure

```
02-inference-benchmarks/
├── src/
│   ├── runners/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract benchmark runner
│   │   ├── vllm_runner.py      # vLLM-specific setup and teardown
│   │   ├── sglang_runner.py    # SGLang-specific setup and teardown
│   │   └── tgi_runner.py       # TGI-specific setup and teardown
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── latency.py          # TTFT, token latency, P50/P95/P99
│   │   ├── throughput.py       # Concurrent request throughput
│   │   └── memory.py           # GPU memory profiling
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py            # Matplotlib/plotly charts
│   └── main.py                 # CLI entrypoint
├── configs/
│   ├── prompts.json            # Standard prompt dataset (short, medium, long)
│   ├── vllm.yaml
│   ├── sglang.yaml
│   └── tgi.yaml
├── results/                    # Raw benchmark outputs (gitignored, samples committed)
│   └── .gitkeep
├── scripts/
│   ├── setup_all_engines.sh    # Install vLLM + SGLang + TGI
│   ├── run_benchmarks.sh       # Run full suite
│   └── generate_report.py      # Produce markdown report from results
├── tests/
│   └── test_metrics.py
└── requirements.txt
```

## Where to Start

### Day 1: Setup all three engines
```bash
# vLLM
pip install vllm

# SGLang
pip install "sglang[all]"

# TGI (Docker)
docker pull ghcr.io/huggingface/text-generation-inference:latest
```

### Day 2: Build the benchmark harness
- `src/runners/base.py` — define the interface: start_server(), send_request(), measure()
- Implement for vLLM first (you already know it from project 01)
- Use async httpx for concurrent request sending

### Day 3: Run benchmarks + collect data
- Standard prompt set: 50 short (< 50 tokens), 50 medium (50-200), 50 long (200+)
- Concurrency levels: 1, 10, 50, 100
- With and without AWQ quantization
- Save raw results as JSON

### Day 4-5: Visualize and write up
- Latency distribution charts (box plots per engine)
- Throughput scaling curves (req/s vs concurrency)
- Memory usage comparison
- Write a clear analysis: "When to choose X over Y"

## Prompt Dataset

Create `configs/prompts.json` with realistic workloads:
- **Short**: "What is 2+2?", classification tasks
- **Medium**: "Summarize this paragraph...", extraction tasks
- **Long**: "Write a detailed analysis of...", generation tasks
- **Code**: "Implement a function that...", code generation

## Success Metrics (what to show in your writeup)

- Clean reproducible benchmark with exact hardware specs
- Clear winner analysis per workload type
- Quantization impact quantified (e.g., "AWQ gives 1.8x throughput at 2% quality loss")
- Actionable recommendation: "Use X for latency-sensitive, Y for throughput"
