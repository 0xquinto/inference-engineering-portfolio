# Inference Stack Benchmarks

Head-to-head evaluation of serving engines on production hardware with real latency percentiles.

## What It Does

This project benchmarks vLLM and SGLang serving the same model under identical conditions — same prompts, same concurrency levels, same hardware. Requests use SSE streaming to measure TTFT and decode speed independently. Results include P50/P95/P99 percentiles, not averages, because tail latency is what determines user experience.

## Key Results

**GPU (Llama 4 Scout 17B-16E on H200 141GB)**

| Engine | Category | Concurrency | P50 (ms) | P99 (ms) | TPS |
|--------|----------|-------------|----------|----------|-----|
| vLLM   | short    | 1           | 218      | 1,913    | 78.5  |
| vLLM   | medium   | 1           | 3,020    | 3,030    | 84.8  |
| vLLM   | medium   | 100         | 4,979    | 5,369    | 50.1  |
| SGLang | short    | 1           | 269      | 1,389    | 93.7  |
| SGLang | medium   | 1           | 2,411    | 2,423    | 106.3 |
| SGLang | medium   | 100         | 3,773    | 4,143    | 65.3  |

**Headline:** SGLang delivers 25-30% higher throughput than vLLM across all workload categories at every concurrency level on this MoE model. SGLang's RadixAttention and chunked prefill give it an edge for MoE scheduling.

## Metrics

| Metric | Description |
|--------|-------------|
| TTFT | Time to first token (ms) — prefill latency |
| TPOT | Time per output token (ms) — decode speed |
| Throughput | Tokens/sec at each concurrency level |
| P50 / P95 / P99 | Latency percentiles under load |
| GPU Memory | VRAM usage (MB) via nvidia-smi |

## Hardware Profiles

| Setting | `gpu` | `local` |
|---------|-------|---------|
| Model | Qwen/Qwen3.5-9B-FP8 | Qwen/Qwen3.5-4B |
| Engines | vLLM, SGLang, TensorRT-LLM | vLLM (via Ollama) |
| Concurrency | 1, 10, 50, 100 | 1, 5, 10 |
| Hardware | H200 141GB / A40 48GB | M4 MacBook (Metal) |

## Usage

```bash
# Run with hardware profile
python -m src.main --profile gpu --engine all
python -m src.main --profile local --engine vllm

# List engines
python -m src.main --list-engines

# Run a single engine
python -m src.main --profile gpu --engine sglang

# Custom config
python -m src.main --config configs/engines.yaml --engine vllm
```

## Project Structure

```
02-inference-benchmarks/
  profiles/
    gpu.yaml            # H200/A40 profile (vLLM + SGLang + TRT-LLM)
    local.yaml          # Apple Silicon profile (Ollama)
  configs/
    engines.yaml        # Default config
    prompts.json        # 16 prompts across 4 categories
  src/
    main.py             # CLI entrypoint
    runners/
      base.py           # Abstract BenchmarkRunner
      vllm_runner.py    # vLLM subprocess runner
      sglang_runner.py  # SGLang subprocess runner
      trtllm_runner.py  # TensorRT-LLM Docker runner
    metrics/
      latency.py        # LatencyTracker (TTFT, TPOT, percentiles)
      throughput.py      # ThroughputCalculator
      memory.py         # nvidia-smi GPU memory parsing
    visualization/
      plots.py          # Latency bar charts, throughput scaling curves
  scripts/
    setup_engines.sh    # GPU environment setup
    setup_local.sh      # Apple Silicon setup
    run_benchmarks.sh   # Full benchmark suite
  results/
    benchmark_combined.json
    latency_c1.png, latency_c10.png, latency_c50.png, latency_c100.png
    throughput_scaling.png
  tests/                # 19 unit tests
  Dockerfile
  requirements.txt
```
