# P07: SLO-Aware Request Scheduling — Design Spec

## Problem

Raw throughput (TPS) is the wrong metric for production LLM serving. A system doing 100 TPS but violating 60% of latency SLOs is worse than one doing 70 TPS with 95% SLO attainment. This project benchmarks how scheduling policy affects **goodput** — the percentage of requests meeting their latency SLO — under realistic mixed workloads.

## What It Proves

Scheduling policy is the lever that separates a demo from a production system. Under uniform load, every policy looks the same. Under mixed, heterogeneous workloads at saturation, FCFS causes head-of-line blocking, priority scheduling helps, and deadline-aware scheduling with admission control maintains high goodput.

## Architecture

```
07-slo-scheduling/
    profiles/
        gpu.yaml              # L40S + vLLM scheduling policies
        local.yaml            # M4 + Ollama + proxy scheduling
    configs/
        scheduling.yaml       # Default config
    src/
        __init__.py
        main.py               # CLI: --profile, --policy, --step
        config.py             # Config loader
        profiles.py           # Profile loader (gpu/local)
        workload.py           # Synthetic workload generator
        scheduler.py          # SLO-aware proxy scheduler
        benchmark.py          # Async benchmarker + SLO measurement
        metrics.py            # Goodput calculation, SLO tracking
        visualize.py          # Charts: goodput vs load, latency CDFs
    results/
    tests/
        __init__.py
        conftest.py
        test_config.py
        test_profiles.py
        test_workload.py
        test_scheduler.py
        test_metrics.py
        test_visualize.py
    scripts/
        setup_gpu.sh          # Install vLLM, pull model
        setup_local.sh        # Install Ollama, pull model
        run_benchmarks.sh     # Start vLLM with scheduling flags, run benchmarks
    requirements.txt
    Dockerfile
```

### Entry Point

```bash
python -m src.main --profile gpu --policy fcfs|priority|slo_aware --step benchmark|visualize|all
```

When `--step visualize` is run standalone, results are loaded from the saved JSON file (same pattern as P04/P05).

### Hardware Profiles

| Profile | Hardware | Engine | Scheduling | Model |
|---------|----------|--------|-----------|-------|
| `gpu` | L40S 48GB | vLLM 0.18.0+ | Engine-native (FCFS, Priority) + proxy (SLO-Aware) | Qwen3.5-9B |
| `local` | M4 MacBook Pro | Ollama | Proxy only (FCFS, SLO-Aware) | Qwen3.5-4B |

**Note:** The Priority policy requires vLLM's `--scheduling-policy priority` flag and `extra_body={"priority": N}` API support. This is GPU-only. Local benchmarks compare FCFS vs SLO-Aware (2 policies instead of 3). The vLLM priority API landed in PR #19057 — verify availability in the vLLM version used.

## Scheduling Policies

Three policies compared (two on local):

### 1. FCFS (Baseline)
vLLM default (`--scheduling-policy fcfs`). Requests processed in arrival order. No awareness of request size or deadlines.

### 2. Priority (GPU only)
vLLM built-in (`--scheduling-policy priority`). Requests assigned priority by estimated output class: short requests get high priority (low numeric value), long requests get low priority. Uses `extra_body={"priority": N}` in the OpenAI API.

### 3. SLO-Aware (Custom)
Lightweight async proxy between client and inference engine:

```
Client requests -> SLO Proxy (reorder + admit) -> vLLM/Ollama -> Response
```

The proxy implements three mechanisms:

1. **Deadline queue** — requests sorted by urgency (arrival_time + SLO_ms). Closest deadline dequeued first.
2. **Admission control** — when queue depth exceeds threshold, reject lowest-urgency requests with HTTP 429 (shed load to protect remaining requests).
3. **Concurrency limiter** — caps in-flight requests to prevent GPU memory pressure from degrading all requests.

Implementation: ~200 lines of async Python using `httpx` + `asyncio.PriorityQueue`. Works with any OpenAI-compatible backend (hardware-agnostic).

## Workload Design

Mixed, heterogeneous workloads where scheduling policy matters:

| Class | Share | Output Tokens | SLO (end-to-end latency) | Example |
|-------|-------|--------------|--------------------------|---------|
| Short | 40% | ~50 | < 2s | Simple Q&A |
| Medium | 40% | ~200 | < 8s | Summarization |
| Long | 20% | ~500+ | < 20s | Code generation |

**SLO metric is end-to-end latency** (arrival to last token), not TTFT. This is critical because output length determines how long a request occupies GPU batch slots — long-generation requests cause head-of-line blocking for subsequent requests. TTFT alone would be insensitive to output length differences.

### Load Levels

QPS = 1, 5, 10, 20 — ramp from underloaded to saturated.

### Workload Generator

`workload.py` produces a stream of requests with:
- Prompt text (varied length to trigger different output lengths)
- Class label (short/medium/long)
- Deadline (arrival_time + class SLO)
- Priority value (for vLLM priority policy)

## Metrics

### Primary: Goodput
Percentage of requests meeting their class-specific end-to-end latency SLO. Reported per-class and overall.

### Secondary
- **End-to-end latency p50/p95/p99** per request class
- **TTFT p50/p95** per request class (for reference, not SLO)
- **Overall throughput** (TPS)
- **Fairness** — ratio of worst-class goodput to best-class goodput (1.0 = perfectly fair)
- **Rejection rate** — % of requests shed by admission control (SLO-aware only)

## Expected Results

| Policy | Goodput @ QPS=1 | Goodput @ QPS=10 | Goodput @ QPS=20 |
|--------|----------------|------------------|------------------|
| FCFS | ~95% | ~60% | ~30% |
| Priority | ~95% | ~75% | ~50% |
| SLO-Aware | ~95% | ~90% | ~80% |

At low load, all policies equivalent. The gap emerges at saturation — FCFS lets long requests block short ones (occupying batch slots for 10+ seconds), Priority helps by serving short requests first, SLO-Aware actively reorders by urgency and sheds load to protect the remaining requests' SLO attainment.

## Visualizations

1. **Goodput vs QPS** — line chart, one line per policy. The key chart showing where policies diverge.
2. **Latency CDF per class** — cumulative distribution of end-to-end latency. SLO threshold marked as vertical line.
3. **Fairness heatmap** — per-class goodput across policies and load levels.

## Testing

### Unit Tests (~35+ tests, no GPU needed)

- `test_config.py` — config loading, profile validation, policy names
- `test_profiles.py` — gpu/local profile loading, invalid profile handling
- `test_workload.py` — class distribution (40/40/20), deadline calculation, prompt generation
- `test_scheduler.py` — deadline queue ordering, admission control rejection, concurrency limiting, starvation prevention
- `test_metrics.py` — goodput calculation edge cases (all pass, all fail, partial, empty)
- `test_visualize.py` — chart generation with mock data

### Key Scheduler Tests

```
# Deadline ordering
requests with deadlines [100ms, 500ms, 200ms] -> dequeued [100, 200, 500]

# Admission control
queue at capacity + new low-urgency request -> rejected with 429

# Concurrency limit
max_concurrent=5, 5 in-flight -> new request waits, doesn't overload GPU
```

## References

- SLAI (UT Austin, 2025): 53% median TTFT reduction via deadline-aware scheduling
- SOLA (Tsinghua, MLSys 2025): 45% to 99% SLO attainment with state-aware scheduling
- SLO-Tuner (2026): black-box online tuning for tail latency guarantees
- vLLM priority scheduling: PR #19057, `--scheduling-policy priority`
- PROSERVE (2025): SlideBatching + GoRouting for gain-oriented scheduling

## Skill Progression

```
Runtime Layer
  01 compress -> 02 engine -> 03 cache -> 04 decode

Agentic Layer
  05 structured output

Systems Layer
  06 cost optimization -> 07 SLO scheduling
```

P07 extends the systems layer: P06 asks "what does each token cost?", P07 asks "how do you guarantee each token arrives on time?"
