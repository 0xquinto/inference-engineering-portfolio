# Inference Engineering Portfolio

Three projects demonstrating inference optimization skills — from smart serving to deep GPU optimization.

## Projects

| # | Project | What it proves | Stack |
|---|---------|---------------|-------|
| 01 | [Agentic Inference Gateway](./01-agentic-gateway/) | Smart multi-model routing for agent workloads | vLLM, FastAPI, SGLang |
| 02 | [Inference Stack Benchmarks](./02-inference-benchmarks/) | Rigorous evaluation of serving frameworks | vLLM, SGLang, TGI |
| 03 | [Speculative Decoding & MoE](./03-speculative-decoding/) | Frontier optimization techniques | vLLM, DeepSpeed |

## Start Here

**Order matters.** Each project builds on skills from the previous one.

```
Project 01 (week 1-2)  →  Project 02 (week 3)  →  Project 03 (week 4-5)
application layer         systems evaluation       deep optimization
```

## GPU Access (cheapest options)

| Provider | GPU | Cost | Good for |
|----------|-----|------|----------|
| RunPod | A10G (24GB) | ~$0.40/hr | Projects 01, 02 (8B models) |
| RunPod | A100 (80GB) | ~$1.50/hr | Project 03, 70B quantized |
| Lambda Cloud | A10G | ~$0.50/hr | Same as above |
| Google Cloud | A100 | Free $300 credits | Best starting point |

## Prerequisites

```bash
# Python 3.10+
python --version

# CUDA toolkit (on your GPU instance)
nvidia-smi

# Core dependencies (install on GPU instance)
pip install vllm fastapi uvicorn httpx pandas matplotlib
```
