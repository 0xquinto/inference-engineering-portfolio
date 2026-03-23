# Inference Engineering Portfolio

Six projects covering the full inference optimization lifecycle — from model compression to token economics. Every project runs on both CUDA GPUs and Apple Silicon via hardware profiles. Same pipeline, same metrics, different hardware.

## Why This Exists

$1 trillion in hardware orders are flowing into inference infrastructure through 2027. A trained model sitting in storage generates zero revenue — inference is where intelligence becomes a business. This portfolio demonstrates the skills that make that infrastructure pay off.

## Projects

| # | Project | What it proves | Key result |
|---|---------|---------------|------------|
| 01 | [Quantization](./01-quantization/) | Compress models without breaking them | 26.7 TPS baseline, INT4/FP8 via llmcompressor + MLX |
| 02 | [Inference Benchmarks](./02-inference-benchmarks/) | Engine selection with data, not opinions | vLLM vs SGLang head-to-head on L40S |
| 03 | [Prefix Caching](./03-prefix-caching/) | Make repeated work free | 2.3x TTFT speedup on GPU, 4 caching scenarios |
| 04 | [Speculative Decoding](./04-speculative-decoding/) | Solve the decode bottleneck | N-gram +14% speedup, MTP +6% on L40S |
| 05 | [Structured Output](./05-structured-output/) | Guarantee agent reliability | 100% validity all backends, ~34 TPS on GPU |
| 06 | [Cost Optimization](./06-cost-optimization/) | Know what each token costs | Edge cascade: 0.8B/4B at $0/M tokens |

## Architecture

```
┌─────────────────────────────────────────────────┐
│          Inference Optimization Pipeline          │
│   quantize → benchmark → cache → speculate →     │
│   constrain → cost-model                         │
├─────────────────────────────────────────────────┤
│             OpenAI-Compatible API                 │
├──────────┬──────────┬────────────────────────────┤
│  vLLM    │  Ollama  │  Any future backend        │
│  (CUDA)  │  (Metal) │  (Groq, TPU, ...)          │
└──────────┴──────────┴────────────────────────────┘
       --profile gpu      --profile local
```

Hardware is a YAML field. The pipeline, metrics, and analysis are the same.

## Skill Progression

```
Runtime Layer
  01 compress the model → 02 choose the engine → 03 optimize the cache → 04 accelerate decode

Agentic Layer
  05 guarantee structured output for tool calling

Systems Layer
  06 model the economics and optimize cost per token
```

## Hardware Profiles

Every project supports `--profile gpu` and `--profile local`:

| Profile | Hardware | Backend | Model | Use case |
|---------|----------|---------|-------|----------|
| `gpu` | L40S 48GB (CUDA) | vLLM 0.18.0 | Qwen3.5-9B | Production benchmarks |
| `local` | M4 MacBook Pro (Metal) | Ollama | Qwen3.5-4B | Development + edge inference |

```bash
python -m src.main --profile gpu     # Production (CUDA)
python -m src.main --profile local   # Apple Silicon (Metal)
python -m src.main --config X.yaml   # Custom configuration
```

## Quick Start

```bash
# Clone and run tests (no GPU needed)
git clone https://github.com/0xquinto/inference-engineering-portfolio.git
cd inference-engineering-portfolio
pip install httpx pyyaml pandas matplotlib pytest pytest-asyncio
python -m pytest 04-speculative-decoding/ -q  # 45 tests

# Run local benchmarks (requires Ollama + model)
ollama pull qwen3.5:4b
cd 04-speculative-decoding
python -m src.main --profile local --method baseline
```

## GPU Access

All GPU benchmarks ran on RunPod with the `runpod-torch-v280` template (CUDA 12.8, PyTorch 2.8.0).

| Provider | GPU | Cost | Projects |
|----------|-----|------|----------|
| RunPod | L40S (48GB) | ~$0.88/hr | 01, 03, 04, 05 |
| RunPod | H200 (141GB) | ~$3.59/hr | 02, 06 (27B tier) |

## Test Coverage

222 tests across all 6 projects, all passing locally without a GPU.

```bash
for proj in 01-quantization 02-inference-benchmarks 03-prefix-caching \
            04-speculative-decoding 05-structured-output 06-cost-optimization; do
    python -m pytest "$proj/" -q
done
```
