# Inference Engineering Portfolio

Three projects demonstrating core inference optimization skills — from model compression to runtime optimization.

## Projects

| # | Project | What it proves | Key technique |
|---|---------|---------------|---------------|
| 01 | [Quantization Pipeline](./01-quantization/) | Compress models and measure quality-speed-memory tradeoffs | GPTQ, AWQ, FP8 quantization |
| 02 | [Inference Stack Benchmarks](./02-inference-benchmarks/) | Rigorous evaluation of serving frameworks on production hardware | vLLM vs SGLang on H200 |
| 03 | [Prefix Caching](./03-prefix-caching/) | KV cache optimization for real inference workloads | Prefix caching, cache-aware routing |

## Skill Progression

```
Project 01 (model optimization)  →  Project 02 (engine evaluation)  →  Project 03 (runtime optimization)
compress the model                  choose the engine                   optimize the serving
```

## GPU Access

| Provider | GPU | Cost | Good for |
|----------|-----|------|----------|
| RunPod | A40 (48GB) | ~$0.75/hr | Projects 01, 03 |
| RunPod | H200 (141GB) | ~$3.59/hr | Project 02 (MoE models) |

## Prerequisites

```bash
python --version  # 3.10+
nvidia-smi        # CUDA toolkit on GPU instance
```

Each project has its own `requirements.txt` and `scripts/setup_gpu.sh` for GPU setup. All tests run locally without a GPU.
