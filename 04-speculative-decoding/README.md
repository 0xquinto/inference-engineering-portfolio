# Speculative Decoding Benchmarks

Benchmark suite comparing speculative decoding strategies in vLLM to measure throughput and latency gains over autoregressive generation.

## What It Does

Autoregressive decoding is memory-bandwidth bound: the GPU computes one token at a time, leaving most arithmetic units idle during each decode step. Speculative decoding addresses this by drafting multiple candidate tokens cheaply, then verifying them in a single forward pass of the target model. This project benchmarks six strategies across increasing request rates, measuring time-to-first-token (TTFT) and decode throughput (tokens/sec).

## Methods

| Method | Type | Description | Availability |
|--------|------|-------------|--------------|
| `baseline` | None | Standard autoregressive decoding | GPU + Local |
| `eagle3` | EAGLE | EAGLE-3 hidden-state speculation using the model's own EAGLE head | GPU only |
| `p_eagle` | EAGLE | Parallel EAGLE -- all draft tokens generated in one forward pass | GPU only |
| `ngram` | N-gram | Suffix matching against the prompt (no extra model needed) | GPU + Local |
| `draft_model` | Draft | Small draft model (Qwen3.5-0.8B) with target-model verification | GPU + Local |
| `mtp` | MTP | Multi-Token Prediction using the model's native MTP heads | GPU only |

## Key Results

**Local Baseline (M4 MacBook Pro, Qwen3.5-4B via Ollama)**

| QPS | TTFT p50 (ms) | TTFT p95 (ms) | Throughput p50 (tok/s) |
|-----|---------------|---------------|------------------------|
| 1   | 38,671        | 87,522        | 5.56                   |
| 5   | 42,900        | 97,595        | 5.07                   |
| 10  | 43,914        | 97,711        | 5.05                   |

Speculative methods (EAGLE-3, P-EAGLE, MTP) require vLLM on GPU and are not available locally. The method-vs-method comparison across all six strategies is the primary deliverable when run on GPU hardware.

## Hardware Profiles

| Profile | Model | Methods | QPS Levels | Requests/Prompt |
|---------|-------|---------|------------|-----------------|
| `gpu`   | Qwen3.5-9B | All 6 | 1, 5, 10, 25, 50 | 10 |
| `local` | Qwen3.5-4B | baseline, ngram, draft_model | 1, 5, 10 | 3 |

## Usage

```bash
# List available methods
python -m src.main --list-methods

# Run full pipeline (benchmark + visualize) with a profile
python -m src.main --profile local
python -m src.main --profile gpu

# Run a single method
python -m src.main --profile gpu --method eagle3

# Run only benchmarks (skip visualization)
python -m src.main --profile gpu --step benchmark

# GPU setup and execution via scripts
bash scripts/setup_gpu.sh
bash scripts/run_benchmarks.sh
```

## Project Structure

```
04-speculative-decoding/
  src/
    main.py          # CLI entrypoint
    config.py        # SpecMethod / SpecConfig dataclasses, YAML loader
    benchmark.py     # Async benchmarker (TTFT, throughput at each QPS)
    methods.py       # Result tracking and speedup calculations
    visualize.py     # TTFT, throughput, and speedup heatmap charts
    profiles.py      # Hardware profile utilities
  profiles/
    gpu.yaml         # 6 methods, Qwen3.5-9B, vLLM
    local.yaml       # 3 methods, Qwen3.5-4B, Ollama
  configs/
    speculative.yaml # Default config (mirrors gpu profile)
  scripts/
    setup_gpu.sh     # Install deps and start vLLM on GPU
    setup_local.sh   # Install deps for local runs
    run_benchmarks.sh
  results/
    speculative_results.json
  tests/
  Dockerfile
  requirements.txt
```
