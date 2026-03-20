# Quantization Pipeline

End-to-end pipeline for quantizing LLMs and measuring the quality-speed tradeoffs.

## What It Does

This project quantizes a base model into multiple formats (INT4, INT8, FP8), then evaluates each variant on quality (perplexity, MMLU) and serving performance (TTFT, throughput) under increasing concurrency. Results are exported as JSON and comparison charts. The pipeline supports both GPU servers (vLLM + llmcompressor) and local Apple Silicon (Ollama + MLX).

## Key Results

**MLX Quantization (M4 MacBook Pro, Qwen3.5-4B)**

| Format | Bits/weight | Size | Compression | Quantization Time |
|--------|------------|------|-------------|-------------------|
| BF16   | 16         | ~8 GB | 1.0x       | -                 |
| INT8   | 8.5        | 4.2 GB | 1.9x      | 4.5s              |
| INT4   | 4.5        | 2.2 GB | 3.6x      | 429s (incl. download) |

**Serving Benchmark (BF16 baseline via Ollama)**

| Concurrency | TTFT (ms) | Throughput (tok/s) |
|-------------|----------:|-----------:|
| 1           | 199       | 26.7       |
| 5           | 17,557    | 12.6       |
| 10          | 38,069    | 7.2        |

## Architecture

The pipeline runs four stages in sequence:

1. **Quantize** -- Convert the base model to each target format (GPTQ/FP8 via llmcompressor on GPU, or MLX on Apple Silicon).
2. **Evaluate** -- Score each variant on perplexity (WikiText-2) and MMLU accuracy.
3. **Benchmark** -- Send concurrent requests to the serving engine and record TTFT and token throughput at each concurrency level.
4. **Visualize** -- Generate comparison bar charts and Pareto frontier plots.

## Hardware Profiles

| Setting              | `gpu` (A100/H100)             | `local` (Apple Silicon)     |
|----------------------|-------------------------------|-----------------------------|
| Model                | Qwen/Qwen3.5-9B              | Qwen/Qwen3.5-4B            |
| Formats              | BF16, W4A16 (GPTQ), FP8      | BF16, INT4 (MLX), INT8 (MLX) |
| Quantization tool    | llmcompressor                 | mlx-lm                     |
| Serving engine       | vLLM (:8010)                  | Ollama (:11434)             |
| Concurrency levels   | 1, 10, 50                     | 1, 5, 10                   |
| Max tokens           | 256                           | 512                         |
| Requests per prompt  | 5                             | 3                           |

## Usage

```bash
# Run full pipeline with a hardware profile
python -m src.main --profile gpu
python -m src.main --profile local

# List available quantization formats
python -m src.main --profile local --list-formats

# Run a single step
python -m src.main --profile gpu --step quantize
python -m src.main --profile gpu --step benchmark

# Run a single format
python -m src.main --profile gpu --format fp8
```

## Project Structure

```
01-quantization/
  profiles/
    gpu.yaml            # A100/H100 profile (vLLM + llmcompressor)
    local.yaml          # Apple Silicon profile (Ollama + MLX)
  configs/
    quantization.yaml   # Default config (same as gpu)
  src/
    main.py             # CLI entrypoint
    config.py           # YAML config loader and dataclasses
    quantize.py         # GPU quantization runner (llmcompressor)
    quantize_mlx.py     # Apple Silicon quantization runner (mlx-lm)
    evaluate.py         # Quality evaluation (perplexity, MMLU)
    benchmark.py        # Concurrent serving benchmarks
    visualize.py        # Chart generation
  scripts/
    run_all.sh          # Full pipeline launcher
    setup_gpu.sh        # GPU environment setup
    setup_local.sh      # Local environment setup
  results/
    quantization_results.json
  tests/
  Dockerfile
  requirements.txt
```
