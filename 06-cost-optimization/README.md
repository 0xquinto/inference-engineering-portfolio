# LLM Cascade Cost Optimization

Model token economics and cascade routing across hardware tiers.

## What It Does

This project models the cost of self-hosted LLM inference and routes queries to
the cheapest model tier that can handle them. A keyword-based cascade classifier
assigns each prompt a complexity level (simple, moderate, complex) and dispatches
it to the appropriate model. The local profile runs all three tiers on-device at
$0/hr; the GPU profile prices cloud instances for breakeven analysis against
commercial APIs (GPT-4o, GPT-4o-mini, Claude Sonnet).

## Key Results (Local Profile -- M4 MacBook)

| Tier | Model | TPS (p50) | $/M Tokens | Monthly Capacity |
|------|-------|-----------|------------|------------------|
| local_small | Qwen3.5-0.8B | 31.5 | $0.00 | ~40.8M tokens |
| local_medium | Qwen3.5-4B | 19.4 | $0.00 | ~25.1M tokens |
| local_large | Qwen3.5-4B (complex) | 8.4 | $0.00 | ~10.9M tokens |

**Blended cascade cost: $0.00/M tokens** -- the entire pipeline runs locally
for free. The GPU profile adds cloud tiers (A40 at $0.75/hr, H200 at $3.59/hr)
to model real $/hr costs and compute breakeven points against API pricing.

## Cascade Routing

The router classifies prompts by scanning for keywords in three buckets:

- **simple** -- "what is", "define", "capital of" -- routed to the smallest model
- **moderate** -- "explain", "compare", "summarize" -- routed to the medium model
- **complex** -- "analyze", "design", "implement" -- routed to the largest model

If no keywords match, the prompt defaults to the complex tier. An optional
quality threshold (`quality_threshold: 0.8`) supports escalation: if the
assigned model scores below the threshold, the request can be re-routed upward.

## Cost Model

Self-hosted cost per million output tokens:

```
cost_per_M = (gpu_cost_per_hour / 3600) / tokens_per_second * 1,000,000
```

The pipeline compares this against API output prices (e.g., GPT-4o at $10/M,
Sonnet at $15/M) and reports savings percentage per tier. Monthly capacity is
calculated as `tps * 3600 * 720 * utilization`.

## Hardware Profiles

**`profiles/gpu.yaml`** -- Three cloud tiers with real pricing:
- small (0.8B) on A40 @ $0.75/hr
- medium (9B) on A40 @ $0.75/hr
- large (27B) on H200 @ $3.59/hr

**`profiles/local.yaml`** -- All-local on Apple Silicon:
- All tiers at $0.00/hr (Ollama on M4 MacBook)
- Same cascade logic, zero marginal cost

## Usage

```bash
# Full pipeline: benchmark, analyze, visualize
python -m src.main --profile local

# Individual steps
python -m src.main --profile local --step benchmark
python -m src.main --profile local --step analyze
python -m src.main --profile local --step visualize

# GPU profile (requires running vLLM instances)
python -m src.main --profile gpu

# List configured model tiers
python -m src.main --profile local --list-models
```

Results are written to `results/cost_optimization_results.json` and charts
are saved as PNGs in `results/`.

## Project Structure

```
06-cost-optimization/
  configs/cost.yaml          # Default config (shared settings)
  profiles/
    gpu.yaml                 # Cloud GPU tiers with $/hr pricing
    local.yaml               # On-device tiers at $0/hr
  src/
    main.py                  # CLI entry point and pipeline orchestration
    cascade.py               # Keyword classifier and cascade router
    cost_model.py            # $/M tokens formula, API comparison, blended cost
    benchmark.py             # Async benchmarker (TPS, TTFT measurement)
    config.py                # YAML config loader and dataclasses
    visualize.py             # Cost charts and distribution plots
  scripts/
    run_benchmarks.sh        # End-to-end benchmark runner
    setup_gpu.sh             # GPU environment setup
    setup_local.sh           # Local Ollama setup
  results/                   # Generated JSON results and PNG charts
  tests/
  Dockerfile
  requirements.txt
```
