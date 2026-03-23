# Structured Output and Constrained Decoding

Benchmarks the throughput and correctness trade-offs of constrained decoding for structured JSON output from LLMs.

## What It Does

In the agentic era, LLM-powered agents depend on guaranteed structured output for reliable tool calling, function routing, and downstream parsing. This project measures whether constrained decoding (grammar-enforced JSON schemas) can deliver 100% schema validity without meaningful throughput loss. It runs controlled experiments across multiple backends, schema complexity levels, and hardware profiles, then visualizes the results.

## Key Results

**GPU (L40S 48GB, Qwen3.5-9B via vLLM 0.18.0)**

| Backend | Schema | Validity | TPS (p50) | Retries |
|---|---|---|---|---|
| xgrammar | simple_json | **100%** | 31.6 | 0 |
| xgrammar | nested_object | **100%** | 36.8 | 0 |
| xgrammar | function_call | **100%** | 31.0 | 0 |
| outlines | simple_json | **100%** | 31.6 | 0 |
| outlines | nested_object | **100%** | 36.9 | 0 |
| outlines | function_call | **100%** | 30.8 | 0 |
| unconstrained | simple_json | **100%** | 31.6 | 0 |
| unconstrained | nested_object | **100%** | 36.9 | 0 |
| unconstrained | function_call | **100%** | 31.2 | 0 |

**GPU finding:** With thinking mode disabled (`enable_thinking: false`), all backends achieve **100% validity** across all schema complexity levels. Throughput is consistent at ~32-37 TPS — constrained decoding adds no overhead on GPU. Without this fix, Qwen3.5-9B wraps output in `<think>` reasoning tags that corrupt guided JSON output.

**Local (M4 MacBook Pro, Qwen3.5-4B via Ollama)**

| Backend | Schema | Validity | TPS (p50) | Retries |
|---|---|---|---|---|
| json_schema | simple_json | **100%** | 5.3 | 0 |
| unconstrained | simple_json | **100%** | 6.3 | 0 |
| json_schema | function_call | 0% | 6.6 | 0 |
| unconstrained | function_call | 0% | 5.5 | 4.0 |

**Local finding:** Both constrained and unconstrained backends achieve 100% validity on simple schemas. The function_call schema remains at 0% — Ollama's constrained decoding does not yet support enum-constrained fields in complex schemas.

**Cross-platform insight:** Reasoning models require explicit thinking-mode control for structured output. Ollama suppresses thinking tokens automatically when `response_format` is set. vLLM requires `chat_template_kwargs: {"enable_thinking": false}` per-request or `--default-chat-template-kwargs` server-side. Without this, guided decoding backends interleave thinking tokens with constrained output, breaking validity.

## Backends

| Backend | Engine | Description |
|---|---|---|
| `xgrammar` | vLLM | XGrammar guided decoding (default in vLLM/SGLang) |
| `outlines` | vLLM | Outlines guided decoding |
| `json_schema` | Ollama | JSON schema constrained decoding |
| `unconstrained` | any | No grammar enforcement (baseline + retry) |

## Schema Complexity Levels

| Schema | Complexity | Description |
|---|---|---|
| `simple_json` | low | Flat object with 3 typed fields |
| `nested_object` | medium | Nested objects with arrays and sub-objects |
| `function_call` | high | Tool-calling schema with enum-constrained function selection |

## Hardware Profiles

| Profile | Engine | Format | Model |
|---|---|---|---|
| `gpu` | vLLM | `guided_json` | Qwen3.5-9B |
| `local` | Ollama | `response_format` | Qwen3.5-4B |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# List available backends and schemas
python -m src.main --list-backends
python -m src.main --list-schemas

# Run full pipeline with a hardware profile
python -m src.main --profile local
python -m src.main --profile gpu

# Run a single backend/schema combination
python -m src.main --profile local --backend json_schema --schema simple_json

# Run only benchmarks (skip visualization)
python -m src.main --profile local --step benchmark

# Run only visualization (from existing results)
python -m src.main --profile local --step visualize

# GPU setup and execution
bash scripts/setup_gpu.sh
bash scripts/run_benchmarks.sh
```

## Project Structure

```
05-structured-output/
    configs/structured.yaml     # Default config
    profiles/
        gpu.yaml                # vLLM + XGrammar/Outlines on GPU
        local.yaml              # Ollama on local machine
    src/
        main.py                 # CLI entrypoint
        config.py               # Config loader (backends, schemas, schema_format)
        schemas.py              # JSON schemas and prompts per complexity level
        benchmark.py            # Async benchmarker (TTFT, TPS, validity)
        visualize.py            # Chart generation
    results/
        structured_output_results.json
        tps_overhead.png
        validity_rate.png
        latency_comparison.png
    scripts/
        setup_gpu.sh
        setup_local.sh
        run_benchmarks.sh
    tests/
    Dockerfile
    requirements.txt
```
