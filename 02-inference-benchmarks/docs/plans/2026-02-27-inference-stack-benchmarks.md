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

## Remaining: Task 10 — Run benchmarks on H200 GPU

### Step 1: Create H200 pod and setup environment

```bash
# Delete old H100 pod
runpodctl pod delete 4umbo2vqiv7prk

# Create H200 pod
runpodctl pod create --name bench-h200 \
  --gpu-id "NVIDIA H200" --gpu-count 1 \
  --image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
  --volume-in-gb 500 \
  --ports "22/tcp,8010/http,8020/http,8030/http"
```

### Step 2: SSH in and prepare

```bash
# Clone repo
cd /workspace
git clone https://github.com/0xquinto/inference-engineering-portfolio.git
cd inference-engineering-portfolio/02-inference-benchmarks

# Setup HF cache on volume (container disk is only 50GB)
mkdir -p /workspace/hf_cache
ln -sf /workspace/hf_cache /root/.cache/huggingface

# Login to HF
pip install huggingface_hub
python -c "from huggingface_hub import login; login(token='<YOUR_HF_TOKEN>')"

# Install shared deps
pip install -r requirements.txt
```

### Step 3: Download FP8 model (~109GB)

```bash
export TMPDIR=/workspace/tmp && mkdir -p $TMPDIR
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Llama-4-Scout-Instruct-FP8')"
```

### Step 4: Install engine venvs (separate due to flashinfer conflict)

```bash
# vLLM venv
python -m venv /workspace/venvs/vllm --system-site-packages
/workspace/venvs/vllm/bin/pip install 'vllm>=0.16' httpx pyyaml

# SGLang venv
python -m venv /workspace/venvs/sglang --system-site-packages
/workspace/venvs/sglang/bin/pip install 'sglang[all]' httpx pyyaml
```

### Step 5: Run benchmarks (one engine at a time)

```bash
bash scripts/run_benchmarks.sh vllm
bash scripts/run_benchmarks.sh sglang
# TRT-LLM if Docker available:
bash scripts/run_benchmarks.sh tensorrt-llm
```

### Step 6: Generate plots and save results

```bash
python -c "from src.visualization import generate_all_plots; import glob; generate_all_plots(glob.glob('results/benchmark_*.json')[0])"
git add results/
git commit -m "data: add benchmark results from H200 GPU"
git push
```

### Step 7: Stop the pod

```bash
runpodctl pod stop <id>
```

---

## Key Lessons Learned

- **Separate venvs required:** vLLM and SGLang pin incompatible flashinfer-python versions
- **RunPod nginx:** Occupies port 8001 — use 8010/8020/8030
- **Container disk is ephemeral:** pip packages, HF login, symlinks lost on restart. Volume at /workspace persists.
- **HF fine-grained tokens:** Don't work for gated repos. Use READ tokens.
- **bitsandbytes int4 broken:** Shape mismatch with Llama 4 MoE in vLLM 0.16. Use pre-quantized FP8 model instead.
- **Model size:** 109B params = ~218GB BF16, ~109GB FP8. H100 (80GB) too small. H200 (141GB) fits FP8 + KV cache.

---

## Latest commit: 7cebaa9
