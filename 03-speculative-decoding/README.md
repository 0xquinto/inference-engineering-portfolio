# 03 — Speculative Decoding & MoE Serving

Frontier inference optimization: speculative decoding for latency, MoE expert profiling for efficiency.

## Why This Project

This is the "wow" project. Most portfolios stop at "I deployed a model with vLLM." This
shows you understand the cutting edge of inference optimization.

Maps directly to roles at: NVIDIA, Anthropic, OpenAI, Together AI, ByteDance.

## What You're Building

### Part A: Speculative Decoding
```
Standard decoding:        [tok1] → [tok2] → [tok3] → [tok4]  (4 forward passes)

Speculative decoding:
  Draft model (small):    [tok1, tok2, tok3, tok4]  (1 pass, speculative)
  Target model (large):   [verify all 4]            (1 pass, parallel verify)
  Result:                 4 tokens in ~2 passes instead of 4
```

### Part B: MoE Expert Profiling
```
┌─────────────────────────────────┐
│     Mixture of Experts Model    │
├────┬────┬────┬────┬────┬────┬───┤
│ E1 │ E2 │ E3 │ E4 │ E5 │ E6 │...│  ← Which experts activate?
└─┬──┴─┬──┴────┴─┬──┴────┴─┬──┴───┘
  │    │         │         │
  ▼    ▼         ▼         ▼
  Profile activation patterns → Optimize expert placement/caching
```

## Directory Structure

```
03-speculative-decoding/
├── src/
│   ├── decoding/
│   │   ├── __init__.py
│   │   ├── speculative.py      # Speculative decoding implementation
│   │   ├── draft_model.py      # Draft model management
│   │   └── verification.py     # Token verification logic
│   ├── profiling/
│   │   ├── __init__.py
│   │   ├── gpu_profiler.py     # CUDA event timing, memory tracking
│   │   └── acceptance_rate.py  # Track speculative acceptance rates
│   ├── moe/
│   │   ├── __init__.py
│   │   ├── expert_profiler.py  # Track which experts activate per input
│   │   ├── serving.py          # MoE-optimized serving config
│   │   └── visualization.py   # Expert activation heatmaps
│   └── main.py
├── configs/
│   ├── speculative.yaml        # Draft/target model pairs
│   └── moe.yaml                # MoE model configs
├── benchmarks/
│   ├── spec_decode_bench.py    # Speculative vs standard decoding
│   └── moe_bench.py            # MoE serving benchmarks
├── scripts/
│   ├── setup_models.sh
│   └── run_experiments.sh
├── tests/
│   ├── test_speculative.py
│   └── test_profiler.py
└── requirements.txt
```

## Where to Start

### Part A: Speculative Decoding (Days 1-5)

#### Day 1: Understand the basics
```bash
# vLLM has built-in speculative decoding support
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --speculative-model meta-llama/Llama-3.1-8B-Instruct \
  --num-speculative-tokens 5 \
  --port 8000
```
Benchmark this against vanilla serving. Measure TTFT and throughput.

#### Day 2-3: Build your own implementation
- Implement speculative decoding from scratch in PyTorch
- Draft model: Llama 3.1 8B → generates N candidate tokens
- Target model: Llama 3.1 70B → verifies candidates in one forward pass
- Track acceptance rate per token position

#### Day 4-5: Optimize and profile
- Tune `num_speculative_tokens` (3, 5, 7, 10) — find the sweet spot
- Profile GPU utilization during draft vs verify phases
- Measure speedup across different prompt types (code, prose, QA)

### Part B: MoE Expert Profiling (Days 6-10)

#### Day 6-7: Serve a MoE model
```bash
# Serve Mixtral or DeepSeek-V3 with vLLM
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --tensor-parallel-size 2 \
  --port 8000
```

#### Day 8-9: Profile expert activations
- Hook into the model's MoE layer to log which experts activate
- Build heatmaps: which experts fire for which input types
- Identify "hot" experts (always active) vs "cold" experts (rarely used)

#### Day 10: Optimization analysis
- Based on activation patterns, propose expert caching strategies
- Measure memory savings from offloading cold experts
- Write up findings with visualizations

## Model Pairs for Speculative Decoding

| Draft Model | Target Model | Expected Speedup |
|-------------|-------------|-----------------|
| Llama 3.1 8B | Llama 3.1 70B | 1.5-2.5x |
| TinyLlama 1.1B | Llama 3.1 8B | 1.3-1.8x |
| Phi-3 Mini | Llama 3.1 70B | 1.4-2.0x |

## Success Metrics

- Speculative decoding speedup quantified with acceptance rate curves
- Token position acceptance rate analysis (position 1 vs 5 vs 10)
- MoE expert activation heatmap across input categories
- GPU utilization profile during draft/verify phases
- Clear writeup: "Here's when speculative decoding helps (and when it doesn't)"
