# SLO-Aware Request Scheduling

Benchmarks how scheduling policy affects goodput (SLO attainment %) under realistic mixed workloads, comparing FCFS, Priority, and a custom deadline-aware proxy scheduler.

## What It Does

Raw throughput (TPS) is the wrong metric for production serving. A system doing 100 TPS but violating 60% of latency SLOs is worse than one doing 70 TPS with 95% SLO attainment. This project measures **goodput** — the percentage of requests meeting their end-to-end latency SLO — across different scheduling policies and load levels. It demonstrates that under mixed workloads at saturation, FCFS causes head-of-line blocking while deadline-aware scheduling with admission control maintains high goodput.

## Key Results

**GPU (L40S 48GB, Qwen3.5-9B via vLLM 0.18.0)**

| Policy | QPS=1 | QPS=5 | QPS=10 | QPS=20 |
|--------|-------|-------|--------|--------|
| FCFS | 20% | 70% | 60% | 70% |
| Priority | 0% | 73% | 60% | 70% |
| **SLO-Aware** | **67%** | **100%** | **100%** | **100%** |

**GPU finding:** The SLO-Aware proxy scheduler achieves **100% goodput** at QPS 5-20 with perfect fairness (1.0), while FCFS and Priority plateau at 60-70%. Deadline-aware reordering with admission control prevents head-of-line blocking from long requests.

**Local (M4 MacBook Pro, Qwen3.5-4B via Ollama)**

| Policy | QPS=1 | QPS=2 | QPS=5 |
|--------|-------|-------|-------|
| FCFS | 60% | 50% | 30% |
| SLO-Aware | 60% | 60% | 50% |

**Local finding:** SLO-Aware maintains 50% goodput at QPS=5 vs FCFS's 30%. The gap widens under saturation. Priority policy is GPU-only (requires vLLM's `--scheduling-policy priority` flag).

**Cross-platform insight:** Scheduling policy is invisible at low load but critical under saturation. The SLO-Aware proxy works identically on both platforms — same deadline queue, same admission control — demonstrating that scheduling logic is hardware-agnostic.

## Scheduling Policies

| Policy | Engine Support | Description |
|--------|---------------|-------------|
| FCFS | vLLM (default) | First-come first-served, no awareness of request size or deadlines |
| Priority | vLLM (GPU only) | Short requests get higher priority via `extra_body={"priority": N}` |
| SLO-Aware | Any (proxy) | Custom deadline queue + admission control + concurrency limiter |

## Workload Design

| Class | Share | Output Tokens | SLO (end-to-end) | Example |
|-------|-------|--------------|------------------|---------|
| Short | 40% | ~50 | < 2s (GPU) / < 5s (local) | Simple Q&A |
| Medium | 40% | ~200 | < 8s (GPU) / < 20s (local) | Summarization |
| Long | 20% | ~500+ | < 20s (GPU) / < 60s (local) | Code generation |

## Hardware Profiles

| Profile | Hardware | Engine | Model |
|---------|----------|--------|-------|
| `gpu` | L40S 48GB | vLLM 0.18.0+ | Qwen3.5-9B |
| `local` | M4 MacBook Pro | Ollama | Qwen3.5-4B |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# List available policies
python -m src.main --list-policies

# Run full pipeline with a hardware profile
python -m src.main --profile local
python -m src.main --profile gpu

# Run a single policy
python -m src.main --profile local --policy slo_aware

# Run only benchmarks (skip visualization)
python -m src.main --profile local --step benchmark

# Run only visualization (from existing results)
python -m src.main --profile local --step visualize
```

## Project Structure

```
07-slo-scheduling/
    configs/scheduling.yaml     # Default config (engine-agnostic)
    profiles/
        gpu.yaml                # vLLM on L40S
        local.yaml              # Ollama on M4
    src/
        main.py                 # CLI entrypoint
        config.py               # Config loader (workload classes, policies, scheduler)
        profiles.py             # Profile loader (gpu/local)
        workload.py             # Synthetic workload generator (short/medium/long)
        scheduler.py            # SLO-aware proxy scheduler (deadline queue)
        benchmark.py            # Async benchmarker (FCFS, Priority, SLO-Aware)
        metrics.py              # Goodput, fairness, latency percentiles
        visualize.py            # Goodput vs QPS, latency CDF, fairness heatmap
    results/
        scheduling_results.json
        goodput_vs_qps.png
        latency_cdf.png
        fairness_heatmap.png
    tests/
    requirements.txt
```
