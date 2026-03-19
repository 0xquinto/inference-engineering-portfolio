# Inference Engineering Portfolio

Six projects covering the full inference optimization lifecycle — from model compression to token economics.

## Projects

| # | Project | What it proves | Key technique |
|---|---------|---------------|---------------|
| 01 | [Quantization Pipeline](./01-quantization/) | Compress models and measure quality-speed-memory tradeoffs | W4A16 INT4, FP8 via llmcompressor |
| 02 | [Inference Stack Benchmarks](./02-inference-benchmarks/) | Rigorous evaluation of serving frameworks on production hardware | vLLM vs SGLang on H200 |
| 03 | [Prefix Caching](./03-prefix-caching/) | KV cache optimization for real inference workloads | Prefix caching, cache-aware routing |
| 04 | [Speculative Decoding](./04-speculative-decoding/) | Accelerate decode with draft-verify methods across QPS levels | EAGLE-3, P-EAGLE, n-gram, draft model |
| 05 | [Structured Output](./05-structured-output/) | Constrained decoding for agentic tool-calling reliability | XGrammar vs Outlines vs unconstrained+retry |
| 06 | [Cost Optimization](./06-cost-optimization/) | Token economics, LLM cascading, and self-hosted vs API breakeven | Cascade routing, cost-per-token modeling |

## Skill Progression

```
Runtime Layer
  01 compress the model → 02 choose the engine → 03 optimize the cache → 04 accelerate decode

Agentic Layer
  05 guarantee structured output for tool calling

Systems Layer
  06 model the economics and optimize cost per token
```

## GPU Access

| Provider | GPU | Cost | Good for |
|----------|-----|------|----------|
| RunPod | A40 (48GB) | ~$0.75/hr | Projects 01, 03, 04, 05, 06 |
| RunPod | H200 (141GB) | ~$3.59/hr | Project 02 (MoE models), 06 (72B tier) |

## Prerequisites

```bash
python --version  # 3.10+
nvidia-smi        # CUDA toolkit on GPU instance
```

Each project has its own `requirements.txt` and `scripts/setup_gpu.sh` for GPU setup. All tests run locally without a GPU.
