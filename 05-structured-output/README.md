# Structured Output and Constrained Decoding

Benchmarks the throughput and correctness trade-offs of constrained decoding for structured JSON output from LLMs.

## What It Does

In the agentic era, LLM-powered agents depend on guaranteed structured output for reliable tool calling, function routing, and downstream parsing. This project measures whether constrained decoding (grammar-enforced JSON schemas) can deliver 100% schema validity without meaningful throughput loss. It runs controlled experiments across multiple backends, schema complexity levels, and hardware profiles, then visualizes the results.

## Key Results

Tested on Qwen3.5-4B with Ollama (`json_schema` backend) on a local machine.

| Backend | Schema | Validity | TPS (p50) | Retries |
|---|---|---|---|---|
| json_schema | simple_json | **100%** | 22.0 | 0 |
| unconstrained | simple_json | 0% | 23.5 | 4.0 |
| json_schema | function_call | 0% | 12.4 | 0 |
| unconstrained | function_call | 0% | 9.3 | 4.0 |

**Headline:** Constrained decoding guarantees correctness on well-supported schemas with only ~6% TPS overhead versus unconstrained generation.

**Note:** The `function_call` schema produced 0% validity even with constraints. This is a model-level limitation (the 4B model cannot reliably follow enum-constrained tool-call schemas), not a framework deficiency.

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
