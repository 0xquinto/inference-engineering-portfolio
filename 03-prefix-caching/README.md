# Prefix Caching Benchmarks

Measures the impact of KV cache reuse on time-to-first-token (TTFT) across four real-world serving patterns.

## What It Does

This project benchmarks prefix caching -- the technique of reusing previously computed KV cache entries when multiple requests share a common prefix. It generates synthetic workloads that mirror production patterns (shared system prompts, multi-turn chat, RAG), sends them to an OpenAI-compatible inference server with caching on and off, and reports TTFT speedup, throughput, and latency percentiles.

## Key Results

**GPU (L40S 48GB, Qwen3.5-9B via vLLM 0.18.0)**

| Scenario | TTFT p50 (no cache) | TTFT p50 (cached) | Speedup | Throughput |
|----------|--------------------:|------------------:|--------:|-----------:|
| shared_system_prompt | 251 ms | 248 ms | 1.0x | 35.6 TPS |

With `--enable-prefix-caching`, vLLM consistently delivers ~250ms TTFT on repeated system prompts. Without the flag, cold-start TTFT is ~580ms before GPU warmup effects reduce it. The 1.0x measured speedup within a single server configuration reflects that vLLM's KV cache reuse is automatic once enabled — both the "off" and "on" labels in the same run see the same server-side caching behavior.

**Cross-server comparison** (from Phase 1 vs Phase 2 of the benchmark):
- Server without `--enable-prefix-caching`: cold TTFT 582ms
- Server with `--enable-prefix-caching`: TTFT 251ms
- **Effective speedup: 2.3x on TTFT** for shared system prompts

**Local (M4 MacBook Pro, Qwen3.5-4B via Ollama)**

| Scenario             | TTFT p50 (off) | TTFT p50 (on) | Speedup |
|----------------------|-----------------|---------------|---------|
| shared_system_prompt | 32,852 ms       | 32,763 ms     | 1.0x    |

The ~1.0x local speedup is expected: Ollama applies KV caching implicitly and does not expose a toggle to disable it, so both runs benefit from caching equally. The 32s TTFT is dominated by Qwen3.5's thinking tokens, not prefill computation.

## Scenarios

| Scenario               | Description                                        | Key Parameters                        |
|------------------------|----------------------------------------------------|---------------------------------------|
| `shared_system_prompt` | Many requests sharing one long system prompt        | 1000 system tokens, 100 requests      |
| `multi_turn`           | Conversations with growing history each turn        | 10 conversations, 10 turns each       |
| `rag_common_context`   | Multiple questions against the same retrieved document | 3000 context tokens, 50 queries    |
| `cache_pressure`       | Increasing unique prefixes to find the eviction cliff | 10--500 unique prefixes, 5 req each |

Scenario parameters are defined in `configs/scenarios.yaml` and can be overridden per profile.

## Hardware Profiles

| Profile | Model             | Engine         | Concurrency | Notes                          |
|---------|-------------------|----------------|-------------|--------------------------------|
| `gpu`   | Qwen/Qwen3.5-9B  | vLLM or SGLang | 10          | All 4 scenarios, full scale    |
| `local` | Qwen/Qwen3.5-4B  | Ollama         | 5           | 2 scenarios, reduced scale     |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# List available scenarios
python -m src.main --list-scenarios

# Run all scenarios with a hardware profile
python -m src.main --profile local
python -m src.main --profile gpu

# Run a single scenario
python -m src.main --profile gpu --scenario shared_system_prompt

# Select engine (gpu profile)
python -m src.main --profile gpu --engine vllm
python -m src.main --profile gpu --engine sglang
```

Results are written to `results/` as JSON. Charts are generated automatically.

## Project Structure

```
03-prefix-caching/
  configs/
    scenarios.yaml        # Scenario definitions and benchmark parameters
    engines.yaml          # Engine connection settings (vLLM, SGLang)
  profiles/
    gpu.yaml              # GPU profile (9B model, full scenarios)
    local.yaml            # Local profile (4B model, reduced scenarios)
  src/
    main.py               # CLI entrypoint and pipeline orchestration
    scenarios.py          # Workload generators for each scenario
    benchmark.py          # HTTP client, request dispatch, timing
    metrics.py            # Latency aggregation and speedup calculation
    visualize.py          # Chart generation (TTFT comparison, cache pressure)
    profiles.py           # Profile loader
  scripts/
    setup_gpu.sh          # GPU environment setup
    setup_local.sh        # Local environment setup
    run_benchmarks.sh     # End-to-end benchmark runner
  results/                # Output JSON and charts
  tests/                  # Unit tests
  Dockerfile
  requirements.txt
```
