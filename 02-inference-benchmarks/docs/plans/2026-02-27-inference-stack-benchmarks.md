# Inference Stack Benchmarks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rigorous, reproducible benchmark of vLLM vs SGLang vs TensorRT-LLM serving Llama 4 Scout 17B-16E on a single H100 GPU, producing publication-ready comparison charts and actionable recommendations.

**Architecture:** Three engine runners implement a shared `BenchmarkRunner` interface. Each runner starts its engine as a subprocess, sends async HTTP requests via httpx at multiple concurrency levels, and collects TTFT, TPOT, throughput, and VRAM metrics. Results are saved as JSON and visualized with matplotlib.

**Tech Stack:** Python 3.11+, httpx (async HTTP), matplotlib/plotly (charts), pyyaml, pandas, pytest, subprocess (engine management). Engines: vLLM v0.16+, SGLang v0.5+, TensorRT-LLM v1.3+ (via Docker). Model: `meta-llama/Llama-4-Scout-17B-16E-Instruct` (MoE, ~109B total params, ~17B active). GPU: RunPod H100 SXM 80GB.

---

## Prerequisites (manual, before starting tasks)

1. ~~**Accept Llama 4 license**~~ — **DONE** (2026-02-27, approved via Maverick grant which covers all Llama 4 models)
2. **Create RunPod H100 pod** — `runpodctl pod create --name bench-h100 --gpuType "NVIDIA H100 80GB HBM3" --gpuCount 1 --imageName runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 --volumeSize 200`
3. **SSH into pod** — `runpodctl pod ssh <id>`
4. **Login to HF on pod** — `pip install huggingface_hub && huggingface-cli login --token <your_token>`

---

### Task 1: Update requirements.txt and project configs

**Files:**
- Modify: `requirements.txt`
- Create: `configs/engines.yaml`
- Modify: `configs/prompts.json` (already good, keep as-is)

**Step 1: Update requirements.txt**

```
vllm>=0.16.0
sglang[all]>=0.5.0
httpx>=0.28.0
pandas>=2.2.0
matplotlib>=3.9.0
plotly>=5.24.0
pyyaml>=6.0
tqdm>=4.67.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
nvidia-ml-py>=12.0.0
```

**Step 2: Create configs/engines.yaml**

```yaml
model:
  name: meta-llama/Llama-4-Scout-17B-16E-Instruct
  type: moe
  active_params: 17B
  total_params: 109B

engines:
  vllm:
    port: 8001
    extra_args:
      - "--max-model-len"
      - "4096"
      - "--dtype"
      - "auto"
      - "--enable-auto-tool-choice"
      - "--tool-call-parser"
      - "llama4"

  sglang:
    port: 8002
    extra_args:
      - "--model-path"
      - "meta-llama/Llama-4-Scout-17B-16E-Instruct"
      - "--port"
      - "8002"
      - "--dtype"
      - "auto"

  tensorrt_llm:
    port: 8003
    docker_image: "nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3"
    extra_args: []

benchmark:
  concurrency_levels: [1, 10, 50, 100]
  warmup_requests: 5
  requests_per_prompt: 3
  max_tokens: 256
  temperature: 0.0

hardware:
  gpu: "NVIDIA H100 80GB HBM3"
  provider: "RunPod"
```

**Step 3: Commit**

```bash
git add requirements.txt configs/engines.yaml
git commit -m "feat: update deps and add engine config for vLLM/SGLang/TRT-LLM"
```

---

### Task 2: Implement metrics module

**Files:**
- Create: `src/metrics/latency.py`
- Create: `src/metrics/throughput.py`
- Create: `src/metrics/memory.py`
- Modify: `src/metrics/__init__.py`
- Create: `tests/test_metrics.py`

**Step 1: Write the failing tests**

```python
# tests/test_metrics.py
"""Tests for metrics collection utilities."""

import pytest
from src.metrics.latency import LatencyTracker
from src.metrics.throughput import ThroughputCalculator
from src.metrics.memory import parse_gpu_memory


class TestLatencyTracker:
    def test_record_and_percentiles(self):
        tracker = LatencyTracker()
        for ms in [100, 200, 300, 400, 500]:
            tracker.record(ms)
        stats = tracker.summary()
        assert stats["count"] == 5
        assert stats["p50"] == 300.0
        assert stats["p95"] >= 400.0
        assert stats["p99"] >= 400.0
        assert stats["mean"] == 300.0

    def test_empty_tracker(self):
        tracker = LatencyTracker()
        stats = tracker.summary()
        assert stats["count"] == 0
        assert stats["p50"] == 0.0

    def test_ttft_and_tpot(self):
        tracker = LatencyTracker()
        tracker.record_request(ttft_ms=50.0, total_ms=500.0, tokens_generated=100)
        stats = tracker.summary()
        assert stats["count"] == 1
        assert stats["ttft_p50"] == 50.0
        assert stats["tpot_p50"] == pytest.approx(4.5, abs=0.1)  # (500-50)/100


class TestThroughputCalculator:
    def test_calculate_throughput(self):
        calc = ThroughputCalculator()
        calc.record_request(total_ms=1000.0, tokens_generated=100)
        calc.record_request(total_ms=2000.0, tokens_generated=200)
        stats = calc.summary()
        assert stats["total_requests"] == 2
        assert stats["total_tokens"] == 300
        assert stats["avg_tokens_per_sec"] > 0

    def test_empty_calculator(self):
        calc = ThroughputCalculator()
        stats = calc.summary()
        assert stats["total_requests"] == 0


class TestGpuMemory:
    def test_parse_gpu_memory_mock(self):
        # Test with mock nvidia-smi output
        mock_output = "45000 MiB, 81920 MiB"
        used, total = parse_gpu_memory(mock_output)
        assert used == 45000
        assert total == 81920
```

**Step 2: Run tests — verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — modules don't exist yet

**Step 3: Implement src/metrics/latency.py**

```python
"""Latency tracking: TTFT, TPOT, and total latency percentiles."""

import statistics
from dataclasses import dataclass, field


class LatencyTracker:
    """Tracks per-request latency metrics and computes percentiles."""

    def __init__(self):
        self._total_ms: list[float] = []
        self._ttft_ms: list[float] = []
        self._tpot_ms: list[float] = []

    def record(self, total_ms: float):
        """Record a simple total latency measurement."""
        self._total_ms.append(total_ms)

    def record_request(self, ttft_ms: float, total_ms: float, tokens_generated: int):
        """Record a full request with TTFT and token-level metrics."""
        self._total_ms.append(total_ms)
        self._ttft_ms.append(ttft_ms)
        if tokens_generated > 0:
            tpot = (total_ms - ttft_ms) / tokens_generated
            self._tpot_ms.append(tpot)

    def summary(self) -> dict:
        result = {
            "count": len(self._total_ms),
            "p50": self._percentile(self._total_ms, 50),
            "p95": self._percentile(self._total_ms, 95),
            "p99": self._percentile(self._total_ms, 99),
            "mean": statistics.mean(self._total_ms) if self._total_ms else 0.0,
        }
        if self._ttft_ms:
            result["ttft_p50"] = self._percentile(self._ttft_ms, 50)
            result["ttft_p95"] = self._percentile(self._ttft_ms, 95)
        if self._tpot_ms:
            result["tpot_p50"] = self._percentile(self._tpot_ms, 50)
            result["tpot_p95"] = self._percentile(self._tpot_ms, 95)
        return result

    def _percentile(self, data: list[float], pct: int) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * pct / 100)
        idx = min(idx, len(sorted_data) - 1)
        return round(sorted_data[idx], 2)
```

**Step 4: Implement src/metrics/throughput.py**

```python
"""Throughput calculation: tokens/sec and requests/sec."""


class ThroughputCalculator:
    """Tracks request and token throughput."""

    def __init__(self):
        self._requests: list[dict] = []

    def record_request(self, total_ms: float, tokens_generated: int):
        self._requests.append({
            "total_ms": total_ms,
            "tokens": tokens_generated,
        })

    def summary(self) -> dict:
        if not self._requests:
            return {"total_requests": 0, "total_tokens": 0, "avg_tokens_per_sec": 0.0}
        total_tokens = sum(r["tokens"] for r in self._requests)
        total_time_s = sum(r["total_ms"] for r in self._requests) / 1000.0
        return {
            "total_requests": len(self._requests),
            "total_tokens": total_tokens,
            "avg_tokens_per_sec": round(total_tokens / total_time_s, 2) if total_time_s > 0 else 0.0,
            "total_time_s": round(total_time_s, 2),
        }
```

**Step 5: Implement src/metrics/memory.py**

```python
"""GPU memory profiling via nvidia-smi."""

import subprocess


def get_gpu_memory() -> tuple[int, int]:
    """Returns (used_mb, total_mb) from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return parse_gpu_memory(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0, 0


def parse_gpu_memory(output: str) -> tuple[int, int]:
    """Parse nvidia-smi memory output like '45000 MiB, 81920 MiB' or '45000, 81920'."""
    parts = output.replace("MiB", "").split(",")
    used = int(parts[0].strip())
    total = int(parts[1].strip())
    return used, total
```

**Step 6: Update src/metrics/__init__.py**

```python
from .latency import LatencyTracker
from .throughput import ThroughputCalculator
from .memory import get_gpu_memory
```

**Step 7: Run tests — verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: all 5 tests PASS

**Step 8: Commit**

```bash
git add src/metrics/ tests/test_metrics.py
git commit -m "feat: add latency, throughput, and memory metrics modules"
```

---

### Task 3: Refactor base runner and implement vLLM runner

**Files:**
- Modify: `src/runners/base.py`
- Create: `src/runners/vllm_runner.py`
- Modify: `src/runners/__init__.py`
- Create: `tests/test_runners.py`

**Step 1: Write the failing tests**

```python
# tests/test_runners.py
"""Tests for benchmark runners (no GPU required — tests runner logic only)."""

import pytest
from src.runners.base import BenchmarkRunner, BenchmarkResult, RequestConfig


def test_benchmark_result_tokens_per_sec():
    r = BenchmarkResult(
        engine="vllm",
        prompt_category="short",
        concurrency=1,
        ttft_ms=50.0,
        total_time_ms=1000.0,
        tokens_generated=100,
        tokens_per_sec=100.0,
        gpu_memory_mb=45000.0,
    )
    assert r.tokens_per_sec == 100.0
    assert r.engine == "vllm"


def test_request_config_defaults():
    cfg = RequestConfig()
    assert cfg.max_tokens == 256
    assert cfg.temperature == 0.0


def test_benchmark_runner_is_abstract():
    with pytest.raises(TypeError):
        BenchmarkRunner()
```

**Step 2: Run tests — verify they fail**

Run: `pytest tests/test_runners.py -v`
Expected: FAIL — RequestConfig not defined

**Step 3: Refactor src/runners/base.py**

```python
"""
Abstract benchmark runner interface.

Each engine (vLLM, SGLang, TensorRT-LLM) implements this interface.
The runner handles: starting the server, sending requests, collecting metrics, and teardown.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx


@dataclass
class RequestConfig:
    max_tokens: int = 256
    temperature: float = 0.0


@dataclass
class BenchmarkResult:
    engine: str
    prompt_category: str
    concurrency: int
    ttft_ms: float
    total_time_ms: float
    tokens_generated: int
    tokens_per_sec: float
    gpu_memory_mb: float


class BenchmarkRunner(ABC):
    """Base class for all inference engine benchmark runners."""

    def __init__(self, engine_name: str, port: int):
        self.engine_name = engine_name
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self._process = None

    @abstractmethod
    async def start_server(self, model: str, extra_args: list[str] | None = None) -> None:
        """Start the inference server."""
        ...

    @abstractmethod
    async def stop_server(self) -> None:
        """Stop the inference server."""
        ...

    async def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the server is ready."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/health", timeout=timeout)
                return resp.status_code == 200
        except httpx.RequestError:
            return False

    async def wait_for_server(self, max_wait: int = 300, interval: int = 5):
        """Wait until the server is ready, up to max_wait seconds."""
        for _ in range(max_wait // interval):
            if await self.health_check():
                return True
            await asyncio.sleep(interval)
        raise TimeoutError(f"{self.engine_name} server did not start within {max_wait}s")

    async def send_request(
        self,
        prompt: str,
        config: RequestConfig | None = None,
    ) -> BenchmarkResult:
        """Send a single chat completion request and measure metrics."""
        cfg = config or RequestConfig()
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "stream": True,
        }

        start = time.perf_counter()
        ttft = None
        tokens = 0

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line.strip() == "data: [DONE]":
                        break
                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000
                    tokens += 1

        total_ms = (time.perf_counter() - start) * 1000
        ttft = ttft or total_ms
        tps = (tokens / (total_ms / 1000)) if total_ms > 0 else 0

        return BenchmarkResult(
            engine=self.engine_name,
            prompt_category="",
            concurrency=1,
            ttft_ms=round(ttft, 2),
            total_time_ms=round(total_ms, 2),
            tokens_generated=tokens,
            tokens_per_sec=round(tps, 2),
            gpu_memory_mb=0,
        )

    async def run_concurrent(
        self,
        prompts: list[str],
        concurrency: int,
        config: RequestConfig | None = None,
    ) -> list[BenchmarkResult]:
        """Send requests at the given concurrency level."""
        sem = asyncio.Semaphore(concurrency)

        async def limited_request(prompt: str) -> BenchmarkResult:
            async with sem:
                return await self.send_request(prompt, config)

        tasks = [limited_request(p) for p in prompts]
        results = await asyncio.gather(*tasks)
        for r in results:
            r.concurrency = concurrency
        return list(results)
```

**Step 4: Implement src/runners/vllm_runner.py**

```python
"""vLLM benchmark runner."""

import asyncio
import subprocess

from .base import BenchmarkRunner


class VllmRunner(BenchmarkRunner):
    """Runs vLLM as a subprocess and benchmarks it."""

    def __init__(self, port: int = 8001):
        super().__init__("vllm", port)

    async def start_server(self, model: str, extra_args: list[str] | None = None) -> None:
        cmd = [
            "vllm", "serve", model,
            "--port", str(self.port),
            *(extra_args or []),
        ]
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        await self.wait_for_server()

    async def stop_server(self) -> None:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
```

**Step 5: Update src/runners/__init__.py**

```python
from .base import BenchmarkRunner, BenchmarkResult, RequestConfig
from .vllm_runner import VllmRunner
```

**Step 6: Run tests — verify they pass**

Run: `pytest tests/test_runners.py -v`
Expected: all 3 tests PASS

**Step 7: Commit**

```bash
git add src/runners/ tests/test_runners.py
git commit -m "feat: refactor base runner and add vLLM runner"
```

---

### Task 4: Implement SGLang and TensorRT-LLM runners

**Files:**
- Create: `src/runners/sglang_runner.py`
- Create: `src/runners/trtllm_runner.py`
- Modify: `src/runners/__init__.py`
- Create: `tests/test_runner_init.py`

**Step 1: Write the failing test**

```python
# tests/test_runner_init.py
"""Tests that all runners can be imported and instantiated."""

from src.runners import VllmRunner
from src.runners.sglang_runner import SglangRunner
from src.runners.trtllm_runner import TrtllmRunner


def test_vllm_runner_init():
    r = VllmRunner(port=8001)
    assert r.engine_name == "vllm"
    assert r.port == 8001


def test_sglang_runner_init():
    r = SglangRunner(port=8002)
    assert r.engine_name == "sglang"
    assert r.port == 8002


def test_trtllm_runner_init():
    r = TrtllmRunner(port=8003)
    assert r.engine_name == "tensorrt-llm"
    assert r.port == 8003
```

**Step 2: Run tests — verify they fail**

Run: `pytest tests/test_runner_init.py -v`
Expected: FAIL — SglangRunner not found

**Step 3: Implement src/runners/sglang_runner.py**

```python
"""SGLang benchmark runner."""

import asyncio
import subprocess

from .base import BenchmarkRunner


class SglangRunner(BenchmarkRunner):
    """Runs SGLang as a subprocess and benchmarks it."""

    def __init__(self, port: int = 8002):
        super().__init__("sglang", port)

    async def start_server(self, model: str, extra_args: list[str] | None = None) -> None:
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", model,
            "--port", str(self.port),
            *(extra_args or []),
        ]
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        await self.wait_for_server()

    async def stop_server(self) -> None:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
```

**Step 4: Implement src/runners/trtllm_runner.py**

```python
"""TensorRT-LLM benchmark runner (via Triton Inference Server Docker container)."""

import asyncio
import subprocess

from .base import BenchmarkRunner


class TrtllmRunner(BenchmarkRunner):
    """Runs TensorRT-LLM via Docker and benchmarks it."""

    def __init__(self, port: int = 8003, docker_image: str = "nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3"):
        super().__init__("tensorrt-llm", port)
        self.docker_image = docker_image
        self._container_id = None

    async def start_server(self, model: str, extra_args: list[str] | None = None) -> None:
        cmd = [
            "docker", "run", "-d", "--gpus", "all",
            "-p", f"{self.port}:8000",
            "--name", "trtllm-bench",
            self.docker_image,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self._container_id = result.stdout.strip()
        await self.wait_for_server(max_wait=600)

    async def stop_server(self) -> None:
        if self._container_id:
            subprocess.run(["docker", "stop", "trtllm-bench"], capture_output=True)
            subprocess.run(["docker", "rm", "trtllm-bench"], capture_output=True)
            self._container_id = None
```

**Step 5: Update src/runners/__init__.py**

```python
from .base import BenchmarkRunner, BenchmarkResult, RequestConfig
from .vllm_runner import VllmRunner
from .sglang_runner import SglangRunner
from .trtllm_runner import TrtllmRunner
```

**Step 6: Run tests — verify they pass**

Run: `pytest tests/test_runner_init.py tests/test_runners.py -v`
Expected: all 6 tests PASS

**Step 7: Commit**

```bash
git add src/runners/ tests/test_runner_init.py
git commit -m "feat: add SGLang and TensorRT-LLM runners"
```

---

### Task 5: Implement CLI entrypoint (main.py)

**Files:**
- Modify: `src/main.py`
- Create: `tests/test_main.py`

**Step 1: Write the failing test**

```python
# tests/test_main.py
"""Tests for CLI entrypoint."""

import subprocess
import sys


def test_help_flag():
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "engine" in result.stdout
    assert "vllm" in result.stdout


def test_list_engines():
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--list-engines"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "vllm" in result.stdout
    assert "sglang" in result.stdout
    assert "tensorrt-llm" in result.stdout
```

**Step 2: Run tests — verify they fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL

**Step 3: Rewrite src/main.py**

```python
"""
Inference Stack Benchmark Suite — CLI entrypoint.

Usage:
    python -m src.main --engine vllm --config configs/engines.yaml
    python -m src.main --engine all --config configs/engines.yaml
    python -m src.main --list-engines
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import yaml

from .runners import VllmRunner, SglangRunner, TrtllmRunner
from .metrics import LatencyTracker, ThroughputCalculator, get_gpu_memory

RUNNERS = {
    "vllm": VllmRunner,
    "sglang": SglangRunner,
    "tensorrt-llm": TrtllmRunner,
}


def load_prompts(path: str = "configs/prompts.json") -> dict[str, list[dict]]:
    with open(path) as f:
        return json.load(f)


def load_config(path: str = "configs/engines.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def run_engine_benchmark(
    engine_name: str,
    model: str,
    prompts: dict[str, list[dict]],
    concurrency_levels: list[int],
    config: dict,
) -> list[dict]:
    """Run full benchmark suite for a single engine."""
    engine_cfg = config["engines"][engine_name.replace("-", "_")]
    runner_cls = RUNNERS[engine_name]
    runner = runner_cls(port=engine_cfg["port"])

    print(f"\n{'='*60}")
    print(f"Benchmarking: {engine_name}")
    print(f"{'='*60}")

    extra_args = engine_cfg.get("extra_args", [])
    await runner.start_server(model, extra_args=extra_args)

    # Get baseline GPU memory
    mem_used, mem_total = get_gpu_memory()
    print(f"GPU Memory: {mem_used} / {mem_total} MB")

    all_results = []
    bench_cfg = config.get("benchmark", {})

    for category, prompt_list in prompts.items():
        flat_prompts = [p["prompt"] for p in prompt_list]

        for concurrency in concurrency_levels:
            print(f"  [{category}] concurrency={concurrency}...", end=" ", flush=True)
            latency = LatencyTracker()
            throughput = ThroughputCalculator()

            # Warmup
            for p in flat_prompts[:2]:
                try:
                    await runner.send_request(p)
                except Exception:
                    pass

            # Benchmark
            results = await runner.run_concurrent(
                flat_prompts * bench_cfg.get("requests_per_prompt", 1),
                concurrency,
            )

            for r in results:
                r.prompt_category = category
                latency.record(r.total_time_ms)
                throughput.record_request(r.total_time_ms, r.tokens_generated)

            lat_summary = latency.summary()
            thr_summary = throughput.summary()
            mem_used, _ = get_gpu_memory()

            result_entry = {
                "engine": engine_name,
                "category": category,
                "concurrency": concurrency,
                "latency": lat_summary,
                "throughput": thr_summary,
                "gpu_memory_mb": mem_used,
            }
            all_results.append(result_entry)
            print(f"p50={lat_summary['p50']}ms, {thr_summary['avg_tokens_per_sec']} tok/s")

    await runner.stop_server()
    return all_results


async def async_main(args):
    config = load_config(args.config)
    prompts = load_prompts(args.prompts)
    model = config["model"]["name"]
    concurrency_levels = config["benchmark"]["concurrency_levels"]

    engines = list(RUNNERS.keys()) if args.engine == "all" else [args.engine]
    all_results = {}

    for engine in engines:
        results = await run_engine_benchmark(engine, model, prompts, concurrency_levels, config)
        all_results[engine] = results

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    output_path = output_dir / f"benchmark_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Inference Stack Benchmarks")
    parser.add_argument("--engine", choices=["vllm", "sglang", "tensorrt-llm", "all"], default="all")
    parser.add_argument("--config", default="configs/engines.yaml")
    parser.add_argument("--prompts", default="configs/prompts.json")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--list-engines", action="store_true", help="List available engines")
    args = parser.parse_args()

    if args.list_engines:
        for name in RUNNERS:
            print(name)
        return

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
```

**Step 4: Run tests — verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: all 2 tests PASS

**Step 5: Commit**

```bash
git add src/main.py tests/test_main.py
git commit -m "feat: implement CLI entrypoint with engine orchestration"
```

---

### Task 6: Implement visualization module

**Files:**
- Create: `src/visualization/plots.py`
- Modify: `src/visualization/__init__.py`
- Create: `tests/test_plots.py`

**Step 1: Write the failing test**

```python
# tests/test_plots.py
"""Tests for visualization (no display needed — tests chart data generation)."""

from src.visualization.plots import prepare_latency_data, prepare_throughput_data


def test_prepare_latency_data():
    results = {
        "vllm": [
            {"engine": "vllm", "category": "short", "concurrency": 1,
             "latency": {"p50": 100, "p95": 200, "p99": 300, "mean": 150, "count": 5}},
        ],
        "sglang": [
            {"engine": "sglang", "category": "short", "concurrency": 1,
             "latency": {"p50": 80, "p95": 160, "p99": 240, "mean": 120, "count": 5}},
        ],
    }
    df = prepare_latency_data(results)
    assert len(df) == 2
    assert "engine" in df.columns
    assert "p50" in df.columns


def test_prepare_throughput_data():
    results = {
        "vllm": [
            {"engine": "vllm", "category": "short", "concurrency": 1,
             "throughput": {"avg_tokens_per_sec": 500, "total_requests": 5}},
            {"engine": "vllm", "category": "short", "concurrency": 10,
             "throughput": {"avg_tokens_per_sec": 4000, "total_requests": 50}},
        ],
    }
    df = prepare_throughput_data(results)
    assert len(df) == 2
    assert "concurrency" in df.columns
```

**Step 2: Run tests — verify they fail**

Run: `pytest tests/test_plots.py -v`
Expected: FAIL

**Step 3: Implement src/visualization/plots.py**

```python
"""Visualization: latency box plots, throughput scaling curves, memory comparison."""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def prepare_latency_data(results: dict) -> pd.DataFrame:
    """Flatten benchmark results into a latency DataFrame."""
    rows = []
    for engine, entries in results.items():
        for entry in entries:
            row = {"engine": engine, "category": entry["category"], "concurrency": entry["concurrency"]}
            row.update(entry["latency"])
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_throughput_data(results: dict) -> pd.DataFrame:
    """Flatten benchmark results into a throughput DataFrame."""
    rows = []
    for engine, entries in results.items():
        for entry in entries:
            row = {"engine": engine, "category": entry["category"], "concurrency": entry["concurrency"]}
            row.update(entry["throughput"])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_latency_comparison(results: dict, output_dir: str = "results/"):
    """Generate latency comparison bar chart (P50/P95 by engine)."""
    df = prepare_latency_data(results)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for conc in df["concurrency"].unique():
        subset = df[df["concurrency"] == conc]
        fig, ax = plt.subplots(figsize=(10, 6))
        engines = subset["engine"].unique()
        x = range(len(engines))
        width = 0.35

        p50_vals = [subset[subset["engine"] == e]["p50"].mean() for e in engines]
        p95_vals = [subset[subset["engine"] == e]["p95"].mean() for e in engines]

        ax.bar([i - width/2 for i in x], p50_vals, width, label="P50", color="#2196F3")
        ax.bar([i + width/2 for i in x], p95_vals, width, label="P95", color="#FF9800")
        ax.set_xlabel("Engine")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Latency Comparison — Concurrency {conc}")
        ax.set_xticks(x)
        ax.set_xticklabels(engines)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_c{conc}.png", dpi=150)
        plt.close()


def plot_throughput_scaling(results: dict, output_dir: str = "results/"):
    """Generate throughput scaling curves (tok/s vs concurrency)."""
    df = prepare_throughput_data(results)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for engine in df["engine"].unique():
        subset = df[df["engine"] == engine].groupby("concurrency")["avg_tokens_per_sec"].mean()
        ax.plot(subset.index, subset.values, marker="o", label=engine, linewidth=2)

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Throughput Scaling: vLLM vs SGLang vs TensorRT-LLM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/throughput_scaling.png", dpi=150)
    plt.close()


def generate_all_plots(results_path: str, output_dir: str = "results/"):
    """Load results JSON and generate all plots."""
    with open(results_path) as f:
        results = json.load(f)
    plot_latency_comparison(results, output_dir)
    plot_throughput_scaling(results, output_dir)
    print(f"Charts saved to {output_dir}/")
```

**Step 4: Update src/visualization/__init__.py**

```python
from .plots import generate_all_plots, plot_latency_comparison, plot_throughput_scaling
```

**Step 5: Run tests — verify they pass**

Run: `pytest tests/test_plots.py -v`
Expected: all 2 tests PASS

**Step 6: Commit**

```bash
git add src/visualization/ tests/test_plots.py
git commit -m "feat: add visualization module with latency and throughput charts"
```

---

### Task 7: Create GPU setup and benchmark scripts

**Files:**
- Create: `scripts/setup_engines.sh`
- Create: `scripts/run_benchmarks.sh`

**Step 1: Create scripts/setup_engines.sh**

```bash
#!/bin/bash
# Setup all three inference engines on a RunPod H100 GPU.
# Run: bash scripts/setup_engines.sh
set -e

echo "=== Setting up Inference Benchmark Suite ==="

# Check GPU
nvidia-smi || { echo "No GPU found."; exit 1; }

# Install Python deps
pip install -r requirements.txt

# Login to HF (token should be set)
huggingface-cli whoami || { echo "Not logged in to HF. Run: huggingface-cli login"; exit 1; }

# Download model
echo "=== Downloading Llama 4 Scout ==="
huggingface-cli download meta-llama/Llama-4-Scout-17B-16E-Instruct
echo "Model downloaded."

# Install vLLM (latest)
echo "=== Installing vLLM ==="
pip install vllm --upgrade
echo "vLLM $(python -c 'import vllm; print(vllm.__version__)')"

# Install SGLang
echo "=== Installing SGLang ==="
pip install "sglang[all]" --upgrade
echo "SGLang installed."

# Pull TensorRT-LLM Docker image
echo "=== Pulling TensorRT-LLM ==="
docker pull nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3 || echo "Docker pull failed — TRT-LLM benchmarks may not work."

echo "=== Setup complete ==="
echo "Run: bash scripts/run_benchmarks.sh"
```

**Step 2: Create scripts/run_benchmarks.sh**

```bash
#!/bin/bash
# Run the full benchmark suite.
# Run: bash scripts/run_benchmarks.sh [engine]
set -e

ENGINE="${1:-all}"
CONFIG="configs/engines.yaml"
PROMPTS="configs/prompts.json"
OUTPUT="results/"

echo "=== Running Inference Stack Benchmarks ==="
echo "Engine: $ENGINE"
echo "Config: $CONFIG"
echo "Output: $OUTPUT"

nvidia-smi

python -m src.main \
    --engine "$ENGINE" \
    --config "$CONFIG" \
    --prompts "$PROMPTS" \
    --output "$OUTPUT"

echo ""
echo "=== Benchmark complete ==="
echo "Results in: $OUTPUT"
echo "Generate plots: python -c \"from src.visualization import generate_all_plots; generate_all_plots('$OUTPUT/benchmark_*.json')\""
```

**Step 3: Commit**

```bash
chmod +x scripts/setup_engines.sh scripts/run_benchmarks.sh
git add scripts/
git commit -m "feat: add GPU setup and benchmark runner scripts"
```

---

### Task 8: Create tests/__init__.py and conftest, run full suite

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create test infrastructure**

```python
# tests/__init__.py
# (empty)
```

```python
# tests/conftest.py
"""Shared test fixtures."""

import pytest
from src.runners.base import RequestConfig


@pytest.fixture
def request_config():
    return RequestConfig(max_tokens=64, temperature=0.0)
```

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: all tests PASS (~12 tests)

**Step 3: Commit**

```bash
git add tests/
git commit -m "feat: add test infrastructure and verify full suite"
```

---

### Task 9: Rewrite README as portfolio showcase

**Files:**
- Modify: `README.md`

**Step 1: Rewrite README.md**

Replace entire contents with a portfolio-ready README:
- Title + one-liner about the benchmark
- Architecture diagram with vLLM / SGLang / TensorRT-LLM on H100
- Metrics table (TTFT, TPOT, throughput, memory)
- How it works (runner interface, concurrent HTTP, streaming metrics)
- Project structure (matching actual files)
- Quick start (setup + run)
- Running tests
- Design decisions (why these 3 engines, why H100, why streaming, why MoE model)
- Remove all planning language, day-by-day roadmap, "Why This Project", "Maps directly to roles at..."

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README as portfolio showcase"
```

---

### Task 10: Run benchmarks on GPU (manual)

This task runs on the RunPod H100 pod, not locally.

**Step 1:** SSH into pod — `runpodctl pod ssh <id>`

**Step 2:** Clone repo and setup
```bash
git clone https://github.com/0xquinto/inference-engineering-portfolio.git
cd inference-engineering-portfolio/02-inference-benchmarks
bash scripts/setup_engines.sh
```

**Step 3:** Run benchmarks per engine (one at a time to avoid OOM)
```bash
bash scripts/run_benchmarks.sh vllm
bash scripts/run_benchmarks.sh sglang
bash scripts/run_benchmarks.sh tensorrt-llm
```

**Step 4:** Generate plots
```bash
python -c "from src.visualization import generate_all_plots; import glob; generate_all_plots(glob.glob('results/benchmark_*.json')[0])"
```

**Step 5:** Copy results to local machine
```bash
# From local:
runpodctl receive results/
```

**Step 6:** Commit results and charts
```bash
git add results/
git commit -m "data: add benchmark results from H100 GPU"
git push
```

**Step 7:** Stop the pod — `runpodctl pod stop <id>`
