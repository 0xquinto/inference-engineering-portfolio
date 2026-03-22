# Structured Output and Constrained Decoding

Benchmarks the throughput and correctness trade-offs of constrained decoding for structured JSON output from LLMs.

## What It Does

In the agentic era, LLM-powered agents depend on guaranteed structured output for reliable tool calling, function routing, and downstream parsing. This project measures whether constrained decoding (grammar-enforced JSON schemas) can deliver 100% schema validity without meaningful throughput loss. It runs controlled experiments across multiple backends, schema complexity levels, and hardware profiles, then visualizes the results.

## Key Results

**GPU (L40S 48GB, Qwen3.5-9B via vLLM 0.18.0)**

| Backend | Schema | Validity | TPS (p50) | Retries |
|---|---|---|---|---|
| xgrammar | simple_json | 0% | 37.9 | 0 |
| xgrammar | nested_object | 0% | 38.0 | 0 |
| xgrammar | function_call | 0% | 37.7 | 0 |
| outlines | simple_json | 0% | 38.0 | 0 |
| outlines | nested_object | 0% | 38.0 | 0 |
| outlines | function_call | 0% | 37.9 | 0 |
| unconstrained | simple_json | 0% | 37.8 | 4.0 |
| unconstrained | nested_object | 0% | 37.9 | 4.0 |
| unconstrained | function_call | 0% | 37.6 | 4.0 |

**GPU finding:** All backends produce 0% validity on GPU. Qwen3.5-9B wraps output in `<think>` reasoning tags before the JSON. Even with `strip_think_tags()` applied, the constrained backends (xgrammar, outlines) embed the thinking tokens inside the guided output, corrupting the JSON structure. Throughput is consistent at ~38 TPS across all backends — constrained decoding adds no overhead on GPU.

**Local (M4 MacBook Pro, Qwen3.5-4B via Ollama)**

| Backend | Schema | Validity | TPS (p50) | Retries |
|---|---|---|---|---|
| json_schema | simple_json | **100%** | 22.0 | 0 |
| unconstrained | simple_json | 0% | 23.5 | 4.0 |
| json_schema | function_call | 0% | 12.4 | 0 |
| unconstrained | function_call | 0% | 9.3 | 4.0 |

**Local finding:** Constrained decoding guarantees correctness on well-supported schemas with only ~6% TPS overhead versus unconstrained generation. Ollama's `json_schema` backend achieves 100% validity on simple schemas.

**Cross-platform insight:** The thinking-mode behavior differs between backends. Ollama suppresses thinking tokens when `response_format` is set, allowing clean JSON output. vLLM's guided_json backends (xgrammar, outlines) interleave thinking tokens with the constrained output, breaking validity. This is a known interaction between reasoning models and constrained decoding — the solution is to disable thinking mode via `chat_template` or use a non-thinking model variant.

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
