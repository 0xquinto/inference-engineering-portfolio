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

**GPU (L40S 48GB, Qwen3.5-9B via vLLM 0.18.0)**

| Method | QPS=1 TPS | QPS=5 TPS | QPS=10 TPS | QPS=25 TPS | QPS=50 TPS | Speedup (QPS=10) |
|--------|-----------|-----------|------------|------------|------------|-------------------|
| baseline | 4.9 | 30.8 | 29.5 | 28.7 | 29.1 | 1.00x |
| ngram | 26.1 | 18.9 | 17.4 | 16.2 | 16.1 | 0.59x |
| mtp | 31.9 | 25.2 | 23.3 | 22.1 | 21.5 | 0.79x |

**Observations:**
- **MTP at QPS=1 beats baseline** (31.9 vs 4.9 TPS) because MTP bypasses the thinking-token bottleneck with a dedicated prediction head, eliminating the 40s TTFT that dominates low-QPS baseline.
- **Both speculative methods lose at higher QPS** — verification overhead exceeds drafting gains when the GPU is already saturated with concurrent requests. This aligns with research showing SD's primary benefit is latency under low-to-medium load, not throughput under high load.
- **N-gram underperforms** (0.59x at QPS=10) because the benchmark prompts are diverse (no repetitive patterns to match), yielding low acceptance rates. N-gram works best on templated/repetitive workloads like summarization or Q&A.
- **Draft model (Qwen3.5-0.8B) is incompatible** — the 0.8B model has hidden_size=1024 vs 9B's 4096, causing a weight shape mismatch in vLLM. The Qwen team's SpecForge framework provides properly trained EAGLE-3 draft models for Qwen.
- **EAGLE-3 / P-EAGLE** require published EAGLE head weights, which are only available for Llama models (yuhuili, RedHatAI on HuggingFace). No Qwen3.5 EAGLE heads exist yet.

**Local (M4 MacBook Pro, Qwen3.5-4B via Ollama)**

| QPS | TTFT p50 (ms) | Throughput p50 (tok/s) |
|-----|---------------|------------------------|
| 1   | 38,671        | 5.6                    |
| 5   | 42,900        | 5.1                    |
| 10  | 43,914        | 5.0                    |

**Cross-platform:** L40S achieves 30.8 TPS vs M4's 5.6 TPS at QPS=5 — a 5.5x throughput advantage. Both platforms show the same pattern: QPS=1 is dominated by thinking tokens (40s+ TTFT), while higher QPS amortizes the overhead.

## Hardware Profiles

| Profile | Model | Methods | QPS Levels | Requests/Prompt |
|---------|-------|---------|------------|-----------------|
| `gpu`   | Qwen3.5-9B | baseline, ngram, mtp (3 tested) | 1, 5, 10, 25, 50 | 10 |
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
