# Portfolio Redesign for Inference Engineering Jobs

**Date:** 2026-03-02
**Goal:** Restructure portfolio to maximize competitiveness for junior inference engineering roles

## Context

Research (Exa deep research + Philip Kiely's "Inference Engineering" book + job postings at NVIDIA, Anthropic, ByteDance, Together AI, Fireworks AI, Baseten, Modal) identified these gaps in the current portfolio:

- No hands-on quantization (used a pre-quantized model, didn't quantize anything)
- No KV cache optimization or prefix caching
- Project 03 (speculative decoding) is an empty scaffold with 0 lines of implementation
- Project 01 (agentic gateway) is more application-layer than inference engineering
- No containerization / production deployment story

## New Portfolio Structure

```
inference-engineering-portfolio/
├── 01-quantization/          # NEW
├── 02-inference-benchmarks/  # KEEP (H200 results already done)
├── 03-prefix-caching/        # NEW
└── README.md
```

Narrative arc: **Model optimization → Engine evaluation → Runtime optimization**

Each project maps to key chapters in "Inference Engineering":
- 01 → Ch 5.1 (Quantization)
- 02 → Ch 4.3/4.5 (Inference Engines / Benchmarking)
- 03 → Ch 5.3 (Caching)

---

## Project 01: Quantization Pipeline & Quality-Speed Tradeoffs

### What it demonstrates
"I can compress models and rigorously measure quality-speed-memory tradeoffs."

### Model
Qwen 2.5 7B Instruct (already familiar with it from project 01, Apache 2.0, ungated)

### Quantization formats to produce
1. **BF16 baseline** — no quantization, reference point
2. **GPTQ INT4** — weight-only, 4-bit integer via AutoGPTQ
3. **AWQ INT4** — weight-only, 4-bit integer via AutoAWQ (activation-aware)
4. **FP8 (E4M3)** — weights + KV cache via llm-compressor or vLLM's built-in

### Quality evaluation
- **Perplexity** on WikiText-2 (simplest quality check)
- **MMLU 5-shot** subset (14 tasks, ~1400 questions — practical accuracy check)
- Report delta from BF16 baseline for each format

### Performance benchmarks
For each quantized format, served via vLLM:
- **TTFT** at concurrency 1, 10, 50
- **Throughput** (tokens/sec) at concurrency 1, 10, 50
- **VRAM usage** (nvidia-smi)
- **Model load time** (cold start)

### Deliverables
- Python scripts for quantization (AutoGPTQ, AutoAWQ, llm-compressor)
- Quality evaluation scripts (perplexity + MMLU)
- Performance benchmark scripts (reuse patterns from project 02)
- Visualization: Pareto frontier chart (quality vs throughput vs VRAM)
- Dockerfile packaging the benchmark suite
- README with analysis: "when to use each format and why"

### GPU requirement
A40 (48GB) or H100 — ~2-3 hours runtime

### Project structure
```
01-quantization/
├── src/
│   ├── quantize.py           # Quantization pipelines (GPTQ, AWQ, FP8)
│   ├── evaluate.py           # Perplexity + MMLU quality eval
│   ├── benchmark.py          # Latency/throughput/VRAM benchmarks via vLLM
│   └── visualize.py          # Pareto frontier and comparison charts
├── configs/
│   └── quantization.yaml     # Model, formats, eval settings
├── results/                  # JSON results + charts (committed)
├── tests/                    # Unit tests (mocked, no GPU needed)
├── scripts/
│   ├── setup_gpu.sh          # Install deps + download model
│   └── run_all.sh            # Quantize → evaluate → benchmark → visualize
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Project 02: Inference Stack Benchmarks (KEEP)

### Changes
- Add Dockerfile
- No other changes needed — H200 results with Llama 4 Scout MoE are already strong

---

## Project 03: Prefix Caching & KV Cache Optimization

### What it demonstrates
"I understand KV cache internals and can measure when prefix caching helps real workloads."

### Engine
vLLM (primary) — has mature `--enable-prefix-caching` flag
SGLang (comparison) — has RadixAttention (automatic prefix caching)

### Model
Qwen 2.5 7B Instruct (consistent across portfolio) or Llama 4 Scout 17B-16E (MoE, consistent with project 02)

### Benchmark scenarios

**Scenario 1: Shared system prompts**
- 100 requests with identical system prompt (500-2000 tokens), varying user messages
- Measure TTFT with prefix caching ON vs OFF
- Expected: dramatic TTFT reduction after first request

**Scenario 2: Multi-turn conversations**
- Simulate 10-turn conversations where each turn adds to the history
- Measure TTFT per turn with caching ON vs OFF
- Expected: TTFT stays flat with caching, grows linearly without

**Scenario 3: RAG with common context**
- 50 different questions against the same 3000-token retrieved document
- Measure throughput (requests/sec) with caching ON vs OFF
- Expected: significant throughput improvement from shared prefix

**Scenario 4: Cache pressure / eviction**
- Gradually increase concurrent unique prefixes beyond cache capacity
- Measure cache hit rate degradation and TTFT impact
- Expected: graceful degradation curve, identify the "cliff"

### Metrics
- **TTFT** (primary) — per-request, p50/p95/p99
- **Cache hit rate** — tracked via engine metrics where available
- **Throughput** (tokens/sec) — at each concurrency level
- **GPU memory** — VRAM usage with caching enabled vs disabled

### Deliverables
- Async benchmark suite with 4 scenarios
- Metrics collection (TTFT waterfall, cache hit curves, memory usage over time)
- Comparison: vLLM prefix caching vs SGLang RadixAttention
- Dockerfile packaging the benchmark suite
- README with analysis: "when prefix caching helps and when it doesn't"

### GPU requirement
Same as project 02 — A40 or H100, ~2-3 hours

### Project structure
```
03-prefix-caching/
├── src/
│   ├── scenarios.py          # 4 benchmark scenarios with prompt generators
│   ├── benchmark.py          # Async benchmark runner (prefix caching on/off)
│   ├── metrics.py            # TTFT, cache hit rate, GPU memory tracking
│   └── visualize.py          # TTFT waterfall, cache hit curves, memory charts
├── configs/
│   ├── scenarios.yaml        # Scenario parameters (prompt lengths, concurrency)
│   └── engines.yaml          # vLLM and SGLang configurations
├── results/                  # JSON results + charts (committed)
├── tests/                    # Unit tests (mocked, no GPU needed)
├── scripts/
│   ├── setup_gpu.sh          # Install engines + download model
│   └── run_benchmarks.sh     # Run all scenarios
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## What gets removed

### Project 01 (Agentic Gateway)
- Archive to a separate repo or delete from this portfolio
- Reason: application-layer project, not inference engineering
- The multi-model routing concept is fine but the heuristic classifier and FastAPI wrapper don't demonstrate inference skills

### Project 03 (Speculative Decoding — empty)
- Delete entirely
- Reason: empty scaffold with aspirational README and 0 implementation
- This actively hurts credibility

---

## Cross-cutting concerns

### Dockerfiles (all projects)
Each project gets a Dockerfile that:
- Uses vLLM's official base image
- Installs project-specific dependencies
- Can run benchmarks end-to-end
- Demonstrates production deployment awareness

### Tests (all projects)
- Unit tests that run locally without GPU (mocked engines)
- Follow existing patterns from projects 01 and 02

### README quality
Each README must have:
- Architecture diagram (ASCII)
- Key results table with real numbers
- "Design Decisions" section explaining technical choices
- Reproducible quick-start instructions

---

## Implementation order

1. Delete project 03 (empty speculative decoding)
2. Archive/move project 01 (agentic gateway)
3. Renumber: benchmarks becomes 02 (stays in place)
4. Build project 01 (quantization) — code locally, run on GPU
5. Build project 03 (prefix caching) — code locally, run on GPU
6. Add Dockerfiles to all 3 projects
7. Update root README.md
8. Final review and commit
