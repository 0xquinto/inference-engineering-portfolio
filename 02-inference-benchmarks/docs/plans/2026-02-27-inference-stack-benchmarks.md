# Inference Stack Benchmarks Implementation Plan

**Goal:** Rigorous, reproducible benchmark of vLLM vs SGLang vs TensorRT-LLM serving Llama 4 Scout 17B-16E on a single H200 GPU, producing publication-ready comparison charts and actionable recommendations.

**Architecture:** Three engine runners implement a shared `BenchmarkRunner` interface. Each runner starts its engine as a subprocess, sends async HTTP requests via httpx at multiple concurrency levels, and collects TTFT, TPOT, throughput, and VRAM metrics. Results are saved as JSON and visualized with matplotlib.

**Tech Stack:** Python 3.11+, httpx (async HTTP), matplotlib/plotly (charts), pyyaml, pandas, pytest, subprocess (engine management). Engines: vLLM v0.16+, SGLang v0.5+, TensorRT-LLM v1.3+ (via Docker). Model: `nvidia/Llama-4-Scout-Instruct-FP8` (MoE, ~109B total params, ~17B active, FP8 precision). GPU: RunPod H200 SXM 141GB HBM3e.

---

## Prerequisites

1. ~~**Accept Llama 4 license**~~ — **DONE** (2026-02-27, approved via Maverick grant which covers all Llama 4 models)
2. ~~**HF token**~~ — **DONE** (READ token saved in memory/huggingface.md)
3. **Create RunPod H200 pod** — `runpodctl pod create --name bench-h200 --gpu-id "NVIDIA H200" --gpu-count 1 --image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 --volume-in-gb 500 --ports "22/tcp,8010/http,8020/http,8030/http"`
4. **Delete old H100 pod** — `runpodctl pod delete 4umbo2vqiv7prk`

---

## Completed Tasks

### Task 1: Update requirements.txt and project configs — DONE
- Created `configs/engines.yaml` (now using FP8 model, H200, ports 8010/8020/8030)
- Updated `requirements.txt` (no engine-specific packages — separate venvs)

### Task 2: Implement metrics module (TDD) — DONE
- `src/metrics/latency.py` — LatencyTracker with TTFT, TPOT, percentiles
- `src/metrics/throughput.py` — ThroughputCalculator with tokens/sec
- `src/metrics/memory.py` — GPU memory via nvidia-smi
- 6 tests pass

### Task 3: Refactor base runner and implement vLLM runner (TDD) — DONE
- `src/runners/base.py` — RequestConfig, BenchmarkResult, BenchmarkRunner ABC
- `src/runners/vllm_runner.py` — VllmRunner with venv binary path support
- 3 tests pass

### Task 4: Implement SGLang and TensorRT-LLM runners (TDD) — DONE
- `src/runners/sglang_runner.py` — SglangRunner with venv binary path support
- `src/runners/trtllm_runner.py` — TrtllmRunner via Docker
- 3 tests pass

### Task 5: Implement CLI entrypoint (TDD) — DONE
- `src/main.py` — Full CLI with --engine, --config, --prompts, --output, --list-engines
- 2 tests pass

### Task 6: Implement visualization module (TDD) — DONE
- `src/visualization/plots.py` — Latency bar charts, throughput scaling curves
- 2 tests pass

### Task 7: Create GPU setup and benchmark scripts — DONE
- `scripts/setup_engines.sh` — Separate venvs for vLLM and SGLang
- `scripts/run_benchmarks.sh` — Selects correct venv per engine

### Task 8: Create test infrastructure and run full suite — DONE
- 16/16 tests pass locally

### Task 9: Rewrite README as portfolio showcase — DONE

---

## Task 10: Run benchmarks on H200 GPU — DONE

- **Pod:** bench-h200 (lzhqms1jtuukso), H200 SXM 141GB, $3.59/hr
- **Model:** RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic (ungated FP8)
- **vLLM 0.16.0:** 83 tok/s (c=1), 49 tok/s (c=100)
- **SGLang 0.5.9:** 103 tok/s (c=1), 65 tok/s (c=100)
- **Winner:** SGLang by 25-35% across all concurrency levels
- GPU memory: vLLM 131GB, SGLang 134GB (of 143GB)
- Results: `results/benchmark_combined.json`, 5 charts in `results/`

---

## Key Lessons Learned

- **Separate venvs required:** vLLM and SGLang pin incompatible flashinfer-python versions
- **RunPod nginx:** Occupies port 8001 — use 8010/8020/8030
- **Container disk is ephemeral:** pip packages, HF login, symlinks lost on restart. Volume at /workspace persists.
- **HF fine-grained tokens:** Don't work for gated repos. Use READ tokens.
- **bitsandbytes int4 broken:** Shape mismatch with Llama 4 MoE in vLLM 0.16. Use pre-quantized FP8 model instead.
- **Model size:** 109B params = ~218GB BF16, ~109GB FP8. H100 (80GB) too small. H200 (141GB) fits FP8 + KV cache.
- **nvidia FP8 model is gated separately:** Use RedHatAI FP8 variant (ungated, same quality).
- **SGLang needs `--context-length`:** Default context length for Llama 4 Scout is huge; causes OOM on KV cache allocation. Must limit to 4096.
- **SGLang needs `libnuma1` and `ninja`:** Not in default RunPod image, install with apt/pip before launching.
- **Subprocess PIPE deadlock:** Don't use `stdout=PIPE` for engine subprocesses — SGLang logs fill the buffer and block. Use `DEVNULL`.
- **SGLang uses `fp8_e4m3` not `fp8`:** The `--kv-cache-dtype` flag differs from vLLM.

---

## All tasks complete
