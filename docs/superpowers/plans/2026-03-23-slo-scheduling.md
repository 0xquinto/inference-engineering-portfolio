# P07: SLO-Aware Request Scheduling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark how scheduling policy affects goodput (SLO attainment %) under mixed workloads, comparing FCFS, Priority, and a custom deadline-aware proxy scheduler.

**Architecture:** Config-driven CLI with hardware profiles (gpu.yaml / local.yaml). Workload generator produces mixed short/medium/long requests. SLO proxy scheduler sits between client and engine, reordering by deadline. Async benchmarker measures end-to-end latency per request and computes goodput.

**Tech Stack:** Python 3.12+, httpx, asyncio, pyyaml, numpy, matplotlib, pandas, pytest

**Spec:** `docs/superpowers/specs/2026-03-23-slo-scheduling-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `07-slo-scheduling/src/__init__.py` | Package marker |
| `07-slo-scheduling/src/main.py` | CLI entry point (--profile, --policy, --step) |
| `07-slo-scheduling/src/config.py` | Config dataclasses + YAML loader |
| `07-slo-scheduling/src/profiles.py` | Profile loader (gpu/local) |
| `07-slo-scheduling/src/workload.py` | Synthetic workload generator (short/medium/long) |
| `07-slo-scheduling/src/scheduler.py` | SLO-aware proxy scheduler (deadline queue + admission control) |
| `07-slo-scheduling/src/benchmark.py` | Async benchmarker (send requests, measure latency + goodput) |
| `07-slo-scheduling/src/metrics.py` | Goodput calculation, SLO tracking, fairness |
| `07-slo-scheduling/src/visualize.py` | Charts (goodput vs QPS, latency CDF, fairness heatmap) |
| `07-slo-scheduling/configs/scheduling.yaml` | Default config |
| `07-slo-scheduling/profiles/gpu.yaml` | L40S + vLLM config |
| `07-slo-scheduling/profiles/local.yaml` | M4 + Ollama config |
| `07-slo-scheduling/tests/__init__.py` | Test package marker |
| `07-slo-scheduling/tests/conftest.py` | Shared fixtures |
| `07-slo-scheduling/tests/test_config.py` | Config loading tests |
| `07-slo-scheduling/tests/test_profiles.py` | Profile loading tests |
| `07-slo-scheduling/tests/test_workload.py` | Workload generator tests |
| `07-slo-scheduling/tests/test_scheduler.py` | Scheduler logic tests |
| `07-slo-scheduling/tests/test_benchmark.py` | Benchmarker unit tests |
| `07-slo-scheduling/tests/test_metrics.py` | Metrics calculation tests |
| `07-slo-scheduling/tests/test_visualize.py` | Visualization tests |
| `07-slo-scheduling/tests/test_main.py` | CLI integration tests |
| `07-slo-scheduling/requirements.txt` | Python dependencies |
| `07-slo-scheduling/Dockerfile` | GPU container image |
| `07-slo-scheduling/scripts/run_benchmarks.sh` | GPU benchmark runner |

---

### Task 1: Project Scaffold + Config

**Files:**
- Create: `07-slo-scheduling/src/__init__.py`
- Create: `07-slo-scheduling/src/config.py`
- Create: `07-slo-scheduling/src/profiles.py`
- Create: `07-slo-scheduling/configs/scheduling.yaml`
- Create: `07-slo-scheduling/profiles/gpu.yaml`
- Create: `07-slo-scheduling/profiles/local.yaml`
- Create: `07-slo-scheduling/requirements.txt`
- Create: `07-slo-scheduling/tests/__init__.py`
- Create: `07-slo-scheduling/tests/conftest.py`
- Create: `07-slo-scheduling/tests/test_config.py`
- Create: `07-slo-scheduling/tests/test_profiles.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p 07-slo-scheduling/{src,configs,profiles,results,tests,scripts}
```

- [ ] **Step 2: Create requirements.txt**

```
httpx>=0.27
pyyaml>=6.0
numpy>=1.26
matplotlib>=3.9
pandas>=2.2
pytest>=8.0
pytest-asyncio>=0.24
```

- [ ] **Step 3: Create configs/scheduling.yaml** (engine-agnostic default)

```yaml
model:
  name: Qwen/Qwen3.5-9B
  type: dense
  params: 9B

workload:
  classes:
    short:
      share: 0.4
      output_tokens: 50
      slo_seconds: 2.0
      prompt: "What is the capital of France? Answer in one sentence."
    medium:
      share: 0.4
      output_tokens: 200
      slo_seconds: 8.0
      prompt: "Summarize the key benefits of renewable energy in 3-4 paragraphs."
    long:
      share: 0.2
      output_tokens: 500
      slo_seconds: 20.0
      prompt: "Write a Python function that implements a binary search tree with insert, delete, and search operations. Include docstrings and type hints."

policies:
  - fcfs
  - slo_aware

benchmark:
  port: 8010
  qps_levels: [1, 5, 10, 20]
  requests_per_qps: 30
  max_tokens: 512
  temperature: 0.0
  warmup_requests: 3

scheduler:
  max_queue_depth: 50
  max_concurrent: 10
```

- [ ] **Step 4: Create profiles/gpu.yaml** (adds engine, model_id, priority policy)

```yaml
model:
  name: Qwen/Qwen3.5-9B
  type: dense
  params: 9B

workload:
  classes:
    short:
      share: 0.4
      output_tokens: 50
      slo_seconds: 2.0
      prompt: "What is the capital of France? Answer in one sentence."
    medium:
      share: 0.4
      output_tokens: 200
      slo_seconds: 8.0
      prompt: "Summarize the key benefits of renewable energy in 3-4 paragraphs."
    long:
      share: 0.2
      output_tokens: 500
      slo_seconds: 20.0
      prompt: "Write a Python function that implements a binary search tree with insert, delete, and search operations. Include docstrings and type hints."

policies:
  - fcfs
  - priority
  - slo_aware

benchmark:
  engine: vllm
  port: 8010
  model_id: "Qwen/Qwen3.5-9B"
  qps_levels: [1, 5, 10, 20]
  requests_per_qps: 30
  max_tokens: 512
  temperature: 0.0
  warmup_requests: 3

scheduler:
  max_queue_depth: 50
  max_concurrent: 10
```

- [ ] **Step 5: Create profiles/local.yaml**

```yaml
model:
  name: Qwen/Qwen3.5-4B
  type: dense
  params: 4B

workload:
  classes:
    short:
      share: 0.4
      output_tokens: 50
      slo_seconds: 5.0
      prompt: "What is the capital of France? Answer in one sentence."
    medium:
      share: 0.4
      output_tokens: 200
      slo_seconds: 20.0
      prompt: "Summarize the key benefits of renewable energy in 3-4 paragraphs."
    long:
      share: 0.2
      output_tokens: 500
      slo_seconds: 60.0
      prompt: "Write a Python function that implements a binary search tree with insert, delete, and search operations. Include docstrings and type hints."

policies:
  - fcfs
  - slo_aware

benchmark:
  engine: ollama
  port: 11434
  model_id: "qwen3.5:4b"
  qps_levels: [1, 2, 5]
  requests_per_qps: 10
  max_tokens: 512
  temperature: 0.0
  warmup_requests: 2

scheduler:
  max_queue_depth: 20
  max_concurrent: 3
```

- [ ] **Step 6: Create src/__init__.py**

Empty file.

- [ ] **Step 7: Create src/profiles.py**

```python
from pathlib import Path

import yaml


def load_profile(name: str, profiles_dir: Path | None = None) -> dict:
    if profiles_dir is None:
        profiles_dir = Path("profiles")
    path = profiles_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
```

- [ ] **Step 8: Create src/config.py**

```python
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class WorkloadClass:
    name: str
    share: float
    output_tokens: int
    slo_seconds: float
    prompt: str

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "WorkloadClass":
        return cls(
            name=name,
            share=data["share"],
            output_tokens=data["output_tokens"],
            slo_seconds=data["slo_seconds"],
            prompt=data["prompt"],
        )


@dataclass
class SchedulerConfig:
    max_queue_depth: int = 50
    max_concurrent: int = 10


@dataclass
class SchedulingConfig:
    model_name: str
    workload_classes: list[WorkloadClass]
    policies: list[str]
    engine: str = "vllm"
    port: int = 8010
    model_id: str = "default"
    qps_levels: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    requests_per_qps: int = 30
    max_tokens: int = 512
    temperature: float = 0.0
    warmup_requests: int = 3
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


def load_config(path: Path) -> SchedulingConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    workload_classes = [
        WorkloadClass.from_dict(name, data)
        for name, data in raw["workload"]["classes"].items()
    ]

    bench = raw.get("benchmark", {})
    sched = raw.get("scheduler", {})

    return SchedulingConfig(
        model_name=raw["model"]["name"],
        workload_classes=workload_classes,
        policies=raw.get("policies", ["fcfs"]),
        engine=bench.get("engine", "vllm"),
        port=bench.get("port", 8010),
        model_id=bench.get("model_id", "default"),
        qps_levels=bench.get("qps_levels", [1, 5, 10, 20]),
        requests_per_qps=bench.get("requests_per_qps", 30),
        max_tokens=bench.get("max_tokens", 512),
        temperature=bench.get("temperature", 0.0),
        warmup_requests=bench.get("warmup_requests", 3),
        scheduler=SchedulerConfig(
            max_queue_depth=sched.get("max_queue_depth", 50),
            max_concurrent=sched.get("max_concurrent", 10),
        ),
    )
```

- [ ] **Step 9: Write test_config.py**

```python
from pathlib import Path

from src.config import WorkloadClass, SchedulerConfig, SchedulingConfig, load_config


class TestWorkloadClass:
    def test_from_dict(self):
        data = {"share": 0.4, "output_tokens": 50, "slo_seconds": 2.0,
                "prompt": "Hello"}
        wc = WorkloadClass.from_dict("short", data)
        assert wc.name == "short"
        assert wc.share == 0.4
        assert wc.output_tokens == 50
        assert wc.slo_seconds == 2.0

    def test_from_dict_all_fields(self):
        data = {"share": 0.2, "output_tokens": 500, "slo_seconds": 20.0,
                "prompt": "Write code"}
        wc = WorkloadClass.from_dict("long", data)
        assert wc.name == "long"
        assert wc.prompt == "Write code"


class TestSchedulerConfig:
    def test_defaults(self):
        sc = SchedulerConfig()
        assert sc.max_queue_depth == 50
        assert sc.max_concurrent == 10

    def test_custom(self):
        sc = SchedulerConfig(max_queue_depth=20, max_concurrent=3)
        assert sc.max_queue_depth == 20


class TestLoadConfig:
    def test_loads_yaml(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "scheduling.yaml")
        assert isinstance(cfg, SchedulingConfig)
        assert cfg.model_name == "Qwen/Qwen3.5-9B"

    def test_workload_classes(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "scheduling.yaml")
        assert len(cfg.workload_classes) == 3
        names = [wc.name for wc in cfg.workload_classes]
        assert "short" in names
        assert "medium" in names
        assert "long" in names

    def test_shares_sum_to_one(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "scheduling.yaml")
        total = sum(wc.share for wc in cfg.workload_classes)
        assert abs(total - 1.0) < 0.01

    def test_policies(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "scheduling.yaml")
        assert "fcfs" in cfg.policies
        assert "slo_aware" in cfg.policies

    def test_qps_levels(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "scheduling.yaml")
        assert cfg.qps_levels == [1, 5, 10, 20]

    def test_scheduler_config(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "scheduling.yaml")
        assert cfg.scheduler.max_queue_depth == 50
        assert cfg.scheduler.max_concurrent == 10
```

- [ ] **Step 10: Write test_profiles.py**

```python
from pathlib import Path

import pytest

from src.profiles import load_profile


class TestLoadProfile:
    def test_load_gpu_profile(self):
        profile = load_profile("gpu", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-9B"
        assert "priority" in profile["policies"]

    def test_load_local_profile(self):
        profile = load_profile("local", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-4B"
        assert "priority" not in profile["policies"]

    def test_invalid_profile_raises(self):
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent", Path(__file__).parent.parent / "profiles")
```

- [ ] **Step 11: Write conftest.py and tests/__init__.py**

`tests/__init__.py`: empty file.

`tests/conftest.py`:
```python
import pytest
from pathlib import Path

import yaml


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "configs" / "scheduling.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_results():
    return {
        "fcfs": {
            1: {"goodput": 0.95, "ttft_p50": 80, "ttft_p95": 200,
                "latency_p50": 1.2, "latency_p95": 3.5, "throughput": 10.0,
                "fairness": 0.9},
            10: {"goodput": 0.60, "ttft_p50": 400, "ttft_p95": 2000,
                 "latency_p50": 5.0, "latency_p95": 15.0, "throughput": 8.0,
                 "fairness": 0.4},
        },
        "slo_aware": {
            1: {"goodput": 0.95, "ttft_p50": 80, "ttft_p95": 200,
                "latency_p50": 1.2, "latency_p95": 3.5, "throughput": 10.0,
                "fairness": 0.9},
            10: {"goodput": 0.90, "ttft_p50": 150, "ttft_p95": 500,
                 "latency_p50": 3.0, "latency_p95": 8.0, "throughput": 9.0,
                 "fairness": 0.85},
        },
    }
```

- [ ] **Step 12: Run tests to verify they pass**

```bash
cd 07-slo-scheduling && python -m pytest tests/test_config.py tests/test_profiles.py -v
```

Expected: All 11 tests pass.

- [ ] **Step 13: Commit**

```bash
git add 07-slo-scheduling/
git commit -m "feat(slo-scheduling): scaffold project with config, profiles, and tests"
```

---

### Task 2: Workload Generator

**Files:**
- Create: `07-slo-scheduling/src/workload.py`
- Create: `07-slo-scheduling/tests/test_workload.py`

- [ ] **Step 1: Write test_workload.py**

```python
import pytest

from src.config import WorkloadClass
from src.workload import WorkloadGenerator, WorkloadRequest


class TestWorkloadRequest:
    def test_fields(self):
        r = WorkloadRequest(
            prompt="hello", request_class="short", slo_seconds=2.0,
            priority=0, deadline=1000.0,
        )
        assert r.request_class == "short"
        assert r.slo_seconds == 2.0
        assert r.priority == 0

    def test_is_short(self):
        r = WorkloadRequest("hi", "short", 2.0, 0, 100.0)
        assert r.request_class == "short"


class TestWorkloadGenerator:
    @pytest.fixture
    def classes(self):
        return [
            WorkloadClass("short", 0.4, 50, 2.0, "short prompt"),
            WorkloadClass("medium", 0.4, 200, 8.0, "medium prompt"),
            WorkloadClass("long", 0.2, 500, 20.0, "long prompt"),
        ]

    def test_generate_count(self, classes):
        gen = WorkloadGenerator(classes)
        requests = gen.generate(100)
        assert len(requests) == 100

    def test_class_distribution(self, classes):
        gen = WorkloadGenerator(classes, seed=42)
        requests = gen.generate(1000)
        short_count = sum(1 for r in requests if r.request_class == "short")
        medium_count = sum(1 for r in requests if r.request_class == "medium")
        long_count = sum(1 for r in requests if r.request_class == "long")
        assert 350 < short_count < 450
        assert 350 < medium_count < 450
        assert 150 < long_count < 250

    def test_deadlines_set(self, classes):
        gen = WorkloadGenerator(classes)
        requests = gen.generate(10)
        for r in requests:
            assert r.deadline > 0
            assert r.slo_seconds > 0

    def test_priority_by_class(self, classes):
        gen = WorkloadGenerator(classes)
        requests = gen.generate(100)
        for r in requests:
            if r.request_class == "short":
                assert r.priority == 0
            elif r.request_class == "medium":
                assert r.priority == 1
            elif r.request_class == "long":
                assert r.priority == 2

    def test_deterministic_with_seed(self, classes):
        gen1 = WorkloadGenerator(classes, seed=42)
        gen2 = WorkloadGenerator(classes, seed=42)
        r1 = gen1.generate(50)
        r2 = gen2.generate(50)
        assert [r.request_class for r in r1] == [r.request_class for r in r2]

    def test_prompts_from_config(self, classes):
        gen = WorkloadGenerator(classes)
        requests = gen.generate(10)
        for r in requests:
            assert len(r.prompt) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd 07-slo-scheduling && python -m pytest tests/test_workload.py -v
```

Expected: FAIL (import errors).

- [ ] **Step 3: Implement workload.py**

```python
import random
import time
from dataclasses import dataclass

from .config import WorkloadClass


PRIORITY_MAP = {"short": 0, "medium": 1, "long": 2}


@dataclass
class WorkloadRequest:
    prompt: str
    request_class: str
    slo_seconds: float
    priority: int
    deadline: float


class WorkloadGenerator:
    def __init__(self, classes: list[WorkloadClass], seed: int | None = None):
        self.classes = classes
        self.rng = random.Random(seed)

    def generate(self, count: int) -> list[WorkloadRequest]:
        population = []
        weights = [wc.share for wc in self.classes]
        chosen = self.rng.choices(self.classes, weights=weights, k=count)
        now = time.monotonic()
        for wc in chosen:
            population.append(WorkloadRequest(
                prompt=wc.prompt,
                request_class=wc.name,
                slo_seconds=wc.slo_seconds,
                priority=PRIORITY_MAP.get(wc.name, 2),
                deadline=now + wc.slo_seconds,
            ))
        return population
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd 07-slo-scheduling && python -m pytest tests/test_workload.py -v
```

Expected: All 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add 07-slo-scheduling/src/workload.py 07-slo-scheduling/tests/test_workload.py
git commit -m "feat(slo-scheduling): add workload generator with class distribution"
```

---

### Task 3: Metrics Module

**Files:**
- Create: `07-slo-scheduling/src/metrics.py`
- Create: `07-slo-scheduling/tests/test_metrics.py`

- [ ] **Step 1: Write test_metrics.py**

```python
import pytest

from src.metrics import RequestResult, compute_goodput, compute_goodput_per_class, compute_fairness, compute_latency_percentiles


class TestRequestResult:
    def test_meets_slo(self):
        r = RequestResult("short", slo_seconds=2.0, latency_seconds=1.5,
                          ttft_seconds=0.1, tokens=50)
        assert r.meets_slo is True

    def test_violates_slo(self):
        r = RequestResult("short", slo_seconds=2.0, latency_seconds=3.0,
                          ttft_seconds=0.1, tokens=50)
        assert r.meets_slo is False

    def test_exact_slo_meets(self):
        r = RequestResult("short", slo_seconds=2.0, latency_seconds=2.0,
                          ttft_seconds=0.1, tokens=50)
        assert r.meets_slo is True


class TestComputeGoodput:
    def test_all_pass(self):
        results = [
            RequestResult("short", 2.0, 1.0, 0.1, 50),
            RequestResult("short", 2.0, 1.5, 0.1, 50),
        ]
        assert compute_goodput(results) == 1.0

    def test_all_fail(self):
        results = [
            RequestResult("short", 2.0, 3.0, 0.1, 50),
            RequestResult("short", 2.0, 5.0, 0.1, 50),
        ]
        assert compute_goodput(results) == 0.0

    def test_partial(self):
        results = [
            RequestResult("short", 2.0, 1.0, 0.1, 50),
            RequestResult("short", 2.0, 3.0, 0.1, 50),
            RequestResult("medium", 8.0, 5.0, 0.1, 200),
            RequestResult("medium", 8.0, 10.0, 0.1, 200),
        ]
        assert compute_goodput(results) == 0.5

    def test_empty(self):
        assert compute_goodput([]) == 0.0

    def test_per_class(self):
        results = [
            RequestResult("short", 2.0, 1.0, 0.1, 50),
            RequestResult("short", 2.0, 3.0, 0.1, 50),
            RequestResult("long", 20.0, 10.0, 0.1, 500),
        ]
        per_class = compute_goodput_per_class(results)
        assert per_class["short"] == 0.5
        assert per_class["long"] == 1.0


class TestComputeFairness:
    def test_perfect_fairness(self):
        per_class = {"short": 1.0, "medium": 1.0, "long": 1.0}
        assert compute_fairness(per_class) == 1.0

    def test_unfair(self):
        per_class = {"short": 1.0, "medium": 0.5, "long": 0.0}
        assert compute_fairness(per_class) == 0.0

    def test_empty(self):
        assert compute_fairness({}) == 0.0


class TestComputeLatencyPercentiles:
    def test_basic(self):
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        p = compute_latency_percentiles(latencies)
        assert p["p50"] == pytest.approx(5.5, abs=0.5)
        assert p["p95"] == pytest.approx(10.0, abs=0.5)
        assert p["p99"] == pytest.approx(10.0, abs=0.5)

    def test_empty(self):
        p = compute_latency_percentiles([])
        assert p["p50"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd 07-slo-scheduling && python -m pytest tests/test_metrics.py -v
```

Expected: FAIL (import errors).

- [ ] **Step 3: Implement metrics.py**

```python
from dataclasses import dataclass


@dataclass
class RequestResult:
    request_class: str
    slo_seconds: float
    latency_seconds: float
    ttft_seconds: float
    tokens: int

    @property
    def meets_slo(self) -> bool:
        return self.latency_seconds <= self.slo_seconds


def compute_goodput(results: list[RequestResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.meets_slo) / len(results)


def compute_goodput_per_class(results: list[RequestResult]) -> dict[str, float]:
    if not results:
        return {}
    by_class: dict[str, list[RequestResult]] = {}
    for r in results:
        by_class.setdefault(r.request_class, []).append(r)
    return {
        cls: sum(1 for r in rs if r.meets_slo) / len(rs)
        for cls, rs in by_class.items()
    }


def compute_fairness(per_class_goodput: dict[str, float]) -> float:
    if not per_class_goodput:
        return 0.0
    values = list(per_class_goodput.values())
    return min(values) / max(values) if max(values) > 0 else 0.0


def compute_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(latencies)
    n = len(s)
    return {
        "p50": s[n // 2],
        "p95": s[int(n * 0.95)] if n >= 20 else s[-1],
        "p99": s[int(n * 0.99)] if n >= 100 else s[-1],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd 07-slo-scheduling && python -m pytest tests/test_metrics.py -v
```

Expected: All 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add 07-slo-scheduling/src/metrics.py 07-slo-scheduling/tests/test_metrics.py
git commit -m "feat(slo-scheduling): add metrics module (goodput, fairness, percentiles)"
```

---

### Task 4: SLO-Aware Proxy Scheduler

**Files:**
- Create: `07-slo-scheduling/src/scheduler.py`
- Create: `07-slo-scheduling/tests/test_scheduler.py`

- [ ] **Step 1: Write test_scheduler.py**

```python
import asyncio
import time

import pytest

from src.scheduler import SLOScheduler, ScheduledRequest
from src.workload import WorkloadRequest


def _make_request(cls="short", slo=2.0, priority=0, deadline=None):
    now = time.monotonic()
    return WorkloadRequest(
        prompt="test", request_class=cls, slo_seconds=slo,
        priority=priority, deadline=deadline or (now + slo),
    )


class TestScheduledRequest:
    def test_ordering_by_deadline(self):
        now = time.monotonic()
        a = ScheduledRequest(_make_request(deadline=now + 1.0))
        b = ScheduledRequest(_make_request(deadline=now + 5.0))
        assert a < b

    def test_equal_deadline_uses_counter(self):
        now = time.monotonic()
        a = ScheduledRequest(_make_request(deadline=now + 1.0))
        b = ScheduledRequest(_make_request(deadline=now + 1.0))
        assert a < b  # a has lower counter (created first)


class TestSLOScheduler:
    @pytest.mark.asyncio
    async def test_enqueue_dequeue_order(self):
        scheduler = SLOScheduler(max_queue_depth=10, max_concurrent=5)
        now = time.monotonic()
        r1 = _make_request(deadline=now + 5.0)
        r2 = _make_request(deadline=now + 1.0)
        r3 = _make_request(deadline=now + 3.0)
        scheduler.enqueue(r1)
        scheduler.enqueue(r2)
        scheduler.enqueue(r3)
        out = await scheduler.dequeue()
        assert out.deadline == r2.deadline  # closest deadline first

    @pytest.mark.asyncio
    async def test_admission_control_rejects(self):
        scheduler = SLOScheduler(max_queue_depth=2, max_concurrent=5)
        now = time.monotonic()
        scheduler.enqueue(_make_request(deadline=now + 1.0))
        scheduler.enqueue(_make_request(deadline=now + 2.0))
        rejected = scheduler.enqueue(_make_request(deadline=now + 10.0))
        assert rejected is True

    @pytest.mark.asyncio
    async def test_admission_accepts_urgent_over_relaxed(self):
        scheduler = SLOScheduler(max_queue_depth=2, max_concurrent=5)
        now = time.monotonic()
        scheduler.enqueue(_make_request(deadline=now + 5.0))
        scheduler.enqueue(_make_request(deadline=now + 10.0))
        rejected = scheduler.enqueue(_make_request(deadline=now + 1.0))
        assert rejected is False
        assert scheduler.queue_size() == 2

    @pytest.mark.asyncio
    async def test_concurrency_limiter(self):
        scheduler = SLOScheduler(max_queue_depth=10, max_concurrent=2)
        assert scheduler.can_dispatch() is True
        scheduler.mark_dispatched()
        scheduler.mark_dispatched()
        assert scheduler.can_dispatch() is False
        scheduler.mark_completed()
        assert scheduler.can_dispatch() is True

    @pytest.mark.asyncio
    async def test_queue_size(self):
        scheduler = SLOScheduler(max_queue_depth=10, max_concurrent=5)
        assert scheduler.queue_size() == 0
        scheduler.enqueue(_make_request())
        assert scheduler.queue_size() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd 07-slo-scheduling && python -m pytest tests/test_scheduler.py -v
```

Expected: FAIL (import errors).

- [ ] **Step 3: Implement scheduler.py**

```python
import asyncio
import heapq
import itertools
from dataclasses import dataclass, field

from .workload import WorkloadRequest


_counter = itertools.count()


@dataclass(order=True)
class ScheduledRequest:
    sort_key: tuple = field(init=False, repr=False)
    request: WorkloadRequest = field(compare=False)
    counter: int = field(default_factory=lambda: next(_counter), compare=False)

    def __post_init__(self):
        self.sort_key = (self.request.deadline, self.counter)

    @property
    def deadline(self):
        return self.request.deadline


class SLOScheduler:
    def __init__(self, max_queue_depth: int = 50, max_concurrent: int = 10):
        self.max_queue_depth = max_queue_depth
        self.max_concurrent = max_concurrent
        self._heap: list[ScheduledRequest] = []
        self._in_flight = 0
        self._event = asyncio.Event()

    def enqueue(self, request: WorkloadRequest) -> bool:
        """Enqueue a request. Returns True if rejected (shed), False if accepted."""
        item = ScheduledRequest(request=request)
        if len(self._heap) < self.max_queue_depth:
            heapq.heappush(self._heap, item)
            self._event.set()
            return False

        # Queue full — reject if new request is less urgent than all queued
        least_urgent = max(self._heap, key=lambda x: x.sort_key)
        if item.sort_key < least_urgent.sort_key:
            # New request is more urgent — evict least urgent
            self._heap.remove(least_urgent)
            heapq.heapify(self._heap)
            heapq.heappush(self._heap, item)
            self._event.set()
            return False

        return True  # Rejected

    async def dequeue(self) -> WorkloadRequest:
        """Wait for and return the most urgent request."""
        while not self._heap:
            self._event.clear()
            await self._event.wait()
        item = heapq.heappop(self._heap)
        return item.request

    def can_dispatch(self) -> bool:
        return self._in_flight < self.max_concurrent

    def mark_dispatched(self):
        self._in_flight += 1

    def mark_completed(self):
        self._in_flight = max(0, self._in_flight - 1)

    def queue_size(self) -> int:
        return len(self._heap)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd 07-slo-scheduling && python -m pytest tests/test_scheduler.py -v
```

Expected: All 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add 07-slo-scheduling/src/scheduler.py 07-slo-scheduling/tests/test_scheduler.py
git commit -m "feat(slo-scheduling): add SLO-aware proxy scheduler with deadline queue"
```

---

### Task 5: Benchmark Harness

**Files:**
- Create: `07-slo-scheduling/src/benchmark.py`
- Create: `07-slo-scheduling/tests/test_benchmark.py`

- [ ] **Step 1: Implement benchmark.py**

```python
import asyncio
import json
import time

import httpx

from .metrics import RequestResult
from .workload import WorkloadRequest
from .scheduler import SLOScheduler


class SchedulingBenchmarker:
    def __init__(
        self, port: int = 8010, max_tokens: int = 512,
        temperature: float = 0.0, model_name: str = "default",
        policy: str = "fcfs",
        scheduler_config: dict | None = None,
    ):
        self.base_url = f"http://localhost:{port}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_name = model_name
        self.policy = policy
        self.scheduler = None
        if policy == "slo_aware" and scheduler_config:
            self.scheduler = SLOScheduler(
                max_queue_depth=scheduler_config.get("max_queue_depth", 50),
                max_concurrent=scheduler_config.get("max_concurrent", 10),
            )

    async def send_request(
        self, request: WorkloadRequest,
    ) -> RequestResult:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.policy == "priority":
            payload["extra_body"] = {"priority": request.priority}

        ttft = 0.0
        tokens = 0
        start = time.perf_counter()

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        if ttft == 0.0:
                            ttft = time.perf_counter() - start
                        tokens += 1

        latency = time.perf_counter() - start
        return RequestResult(
            request_class=request.request_class,
            slo_seconds=request.slo_seconds,
            latency_seconds=latency,
            ttft_seconds=ttft,
            tokens=tokens,
        )

    async def run_at_qps(
        self, requests: list[WorkloadRequest], qps: float,
    ) -> list[RequestResult]:
        interval = 1.0 / qps if qps > 0 else 0.0

        if self.policy == "slo_aware" and self.scheduler:
            return await self._run_scheduled(requests, interval)

        return await self._run_direct(requests, interval)

    async def _run_direct(
        self, requests: list[WorkloadRequest], interval: float,
    ) -> list[RequestResult]:
        tasks = []
        for i, req in enumerate(requests):
            if i > 0 and interval > 0:
                await asyncio.sleep(interval)
            tasks.append(asyncio.create_task(self.send_request(req)))
        return list(await asyncio.gather(*tasks))

    async def _run_scheduled(
        self, requests: list[WorkloadRequest], interval: float,
    ) -> list[RequestResult]:
        results: list[RequestResult] = []
        sem = asyncio.Semaphore(self.scheduler.max_concurrent)
        done_event = asyncio.Event()
        rejected_count = 0
        produced_count = 0

        async def producer():
            nonlocal rejected_count, produced_count
            for i, req in enumerate(requests):
                if i > 0 and interval > 0:
                    await asyncio.sleep(interval)
                if self.scheduler.enqueue(req):
                    rejected_count += 1
                produced_count += 1
            done_event.set()

        async def dispatch_one(req: WorkloadRequest):
            async with sem:
                result = await self.send_request(req)
                results.append(result)

        async def consumer():
            tasks = []
            while True:
                if self.scheduler.queue_size() > 0:
                    req = await self.scheduler.dequeue()
                    tasks.append(asyncio.create_task(dispatch_one(req)))
                elif done_event.is_set() and self.scheduler.queue_size() == 0:
                    break
                else:
                    await asyncio.sleep(0.01)
            await asyncio.gather(*tasks)

        await asyncio.gather(producer(), consumer())
        return results
```

- [ ] **Step 2: Write test_benchmark.py**

```python
from src.benchmark import SchedulingBenchmarker


class TestSchedulingBenchmarker:
    def test_init_default(self):
        b = SchedulingBenchmarker(port=8010, model_name="test")
        assert b.base_url == "http://localhost:8010"
        assert b.policy == "fcfs"
        assert b.scheduler is None

    def test_init_slo_aware_creates_scheduler(self):
        b = SchedulingBenchmarker(
            port=8010, model_name="test", policy="slo_aware",
            scheduler_config={"max_queue_depth": 20, "max_concurrent": 5},
        )
        assert b.scheduler is not None
        assert b.scheduler.max_queue_depth == 20
        assert b.scheduler.max_concurrent == 5

    def test_init_priority_no_scheduler(self):
        b = SchedulingBenchmarker(port=8010, model_name="test", policy="priority")
        assert b.scheduler is None
```

- [ ] **Step 3: Run tests**

```bash
cd 07-slo-scheduling && python -m pytest tests/test_benchmark.py -v
```

Expected: All 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add 07-slo-scheduling/src/benchmark.py 07-slo-scheduling/tests/test_benchmark.py
git commit -m "feat(slo-scheduling): add async benchmark harness with scheduling support"
```

---

### Task 6: Visualization

**Files:**
- Create: `07-slo-scheduling/src/visualize.py`
- Create: `07-slo-scheduling/tests/test_visualize.py`

- [ ] **Step 1: Write test_visualize.py**

```python
import os
import tempfile

import matplotlib
matplotlib.use("Agg")

from src.visualize import plot_goodput_vs_qps, plot_latency_cdf, plot_fairness_heatmap


class TestPlotGoodputVsQps:
    def test_creates_file(self):
        data = {
            "fcfs": {1: {"goodput": 0.95}, 10: {"goodput": 0.60}},
            "slo_aware": {1: {"goodput": 0.95}, 10: {"goodput": 0.90}},
        }
        with tempfile.TemporaryDirectory() as d:
            plot_goodput_vs_qps(data, d)
            assert os.path.exists(os.path.join(d, "goodput_vs_qps.png"))


class TestPlotLatencyCdf:
    def test_creates_file(self):
        data = {
            "fcfs": {10: {"latencies": [1.0, 2.0, 3.0, 5.0, 8.0]}},
            "slo_aware": {10: {"latencies": [1.0, 1.5, 2.0, 2.5, 3.0]}},
        }
        slos = {"short": 2.0, "medium": 8.0, "long": 20.0}
        with tempfile.TemporaryDirectory() as d:
            plot_latency_cdf(data, slos, d)
            assert os.path.exists(os.path.join(d, "latency_cdf.png"))


class TestPlotFairnessHeatmap:
    def test_creates_file(self):
        data = {
            "fcfs": {1: {"fairness": 0.9}, 10: {"fairness": 0.4}},
            "slo_aware": {1: {"fairness": 0.9}, 10: {"fairness": 0.85}},
        }
        with tempfile.TemporaryDirectory() as d:
            plot_fairness_heatmap(data, d)
            assert os.path.exists(os.path.join(d, "fairness_heatmap.png"))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd 07-slo-scheduling && python -m pytest tests/test_visualize.py -v
```

- [ ] **Step 3: Implement visualize.py**

```python
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_goodput_vs_qps(data: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, qps_data in data.items():
        qps_vals = sorted(int(q) for q in qps_data.keys())
        goodput_vals = [qps_data[str(q) if str(q) in qps_data else q]["goodput"] * 100 for q in qps_vals]
        ax.plot(qps_vals, goodput_vals, marker="o", label=policy.upper())
    ax.set_xlabel("QPS (Queries Per Second)")
    ax.set_ylabel("Goodput (% meeting SLO)")
    ax.set_title("Goodput vs Load by Scheduling Policy")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "goodput_vs_qps.png"), dpi=150)
    plt.close(fig)


def plot_latency_cdf(data: dict, slos: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, qps_data in data.items():
        for qps, metrics in qps_data.items():
            latencies = sorted(metrics.get("latencies", []))
            if not latencies:
                continue
            cdf = np.arange(1, len(latencies) + 1) / len(latencies)
            ax.plot(latencies, cdf, label=f"{policy.upper()} QPS={qps}")
    for cls, slo in slos.items():
        ax.axvline(x=slo, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("End-to-End Latency (seconds)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency Distribution by Policy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "latency_cdf.png"), dpi=150)
    plt.close(fig)


def plot_fairness_heatmap(data: dict, output_dir: str):
    policies = sorted(data.keys())
    qps_vals = sorted({q for p in data.values() for q in p.keys()})
    matrix = []
    for policy in policies:
        row = [data[policy].get(q, {}).get("fairness", 0) for q in qps_vals]
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(qps_vals)))
    ax.set_xticklabels([str(q) for q in qps_vals])
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels([p.upper() for p in policies])
    ax.set_xlabel("QPS")
    ax.set_title("Fairness (min/max class goodput)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fairness_heatmap.png"), dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd 07-slo-scheduling && python -m pytest tests/test_visualize.py -v
```

Expected: All 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add 07-slo-scheduling/src/visualize.py 07-slo-scheduling/tests/test_visualize.py
git commit -m "feat(slo-scheduling): add visualization (goodput, latency CDF, fairness)"
```

---

### Task 7: CLI Entry Point

**Files:**
- Create: `07-slo-scheduling/src/main.py`
- Create: `07-slo-scheduling/tests/test_main.py`

- [ ] **Step 1: Implement main.py**

```python
import argparse
import asyncio
import json
from pathlib import Path

from .config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="SLO-Aware Request Scheduling Benchmarks",
    )
    parser.add_argument("--config", default="configs/scheduling.yaml",
                        help="Config file path")
    parser.add_argument("--profile", choices=["gpu", "local"],
                        help="Hardware profile (gpu or local)")
    parser.add_argument("--output", default="results/",
                        help="Output directory for results")
    parser.add_argument("--policy",
                        help="Run only this policy (default: all from config)")
    parser.add_argument("--list-policies", action="store_true",
                        help="List available scheduling policies")
    parser.add_argument("--step", choices=["benchmark", "visualize", "all"],
                        default="all", help="Which pipeline step to run")
    args = parser.parse_args()

    if args.profile:
        config_path = Path(f"profiles/{args.profile}.yaml")
    else:
        config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.list_policies:
        print("Available policies:")
        for p in cfg.policies:
            print(f"  {p}")
        return

    asyncio.run(async_main(args, cfg))


async def async_main(args, cfg):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    results_file = output / "scheduling_results.json"

    policies = cfg.policies
    if args.policy:
        if args.policy not in policies:
            print(f"Error: policy '{args.policy}' not in config")
            return
        policies = [args.policy]

    all_results = {}
    if results_file.exists():
        try:
            with open(results_file) as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass

    if args.step in ("benchmark", "all"):
        print("=== Step 1: Scheduling Benchmarks ===")
        from .benchmark import SchedulingBenchmarker
        from .workload import WorkloadGenerator
        from .metrics import compute_goodput, compute_goodput_per_class, compute_fairness, compute_latency_percentiles

        for policy in policies:
            print(f"\n  Policy: {policy}")
            all_results[policy] = {}

            benchmarker = SchedulingBenchmarker(
                port=cfg.port, max_tokens=cfg.max_tokens,
                temperature=cfg.temperature, model_name=cfg.model_id,
                policy=policy,
                scheduler_config={
                    "max_queue_depth": cfg.scheduler.max_queue_depth,
                    "max_concurrent": cfg.scheduler.max_concurrent,
                },
            )

            gen = WorkloadGenerator(cfg.workload_classes, seed=42)

            for qps in cfg.qps_levels:
                requests = gen.generate(cfg.requests_per_qps)
                print(f"    QPS={qps} ({len(requests)} requests)...")

                results = await benchmarker.run_at_qps(requests, qps)

                per_class = compute_goodput_per_class(results)
                overall = compute_goodput(results)
                fairness = compute_fairness(per_class)
                latencies = [r.latency_seconds for r in results]
                ttfts = [r.ttft_seconds for r in results]
                lat_pct = compute_latency_percentiles(latencies)
                ttft_pct = compute_latency_percentiles(ttfts)

                all_results[policy][qps] = {
                    "goodput": overall,
                    "goodput_per_class": per_class,
                    "fairness": fairness,
                    "latency_p50": lat_pct["p50"],
                    "latency_p95": lat_pct["p95"],
                    "ttft_p50": ttft_pct["p50"],
                    "ttft_p95": ttft_pct["p95"],
                    "latencies": latencies,
                    "total_requests": len(requests),
                }

                print(
                    f"      goodput={overall:.0%}, fairness={fairness:.2f}, "
                    f"lat_p50={lat_pct['p50']:.1f}s, lat_p95={lat_pct['p95']:.1f}s"
                )

    if args.step in ("visualize", "all"):
        print("\n=== Step 2: Visualization ===")
        if not all_results and results_file.exists():
            with open(results_file) as f:
                all_results = json.load(f)
        from .visualize import plot_goodput_vs_qps, plot_latency_cdf, plot_fairness_heatmap
        slos = {wc.name: wc.slo_seconds for wc in cfg.workload_classes}
        plot_goodput_vs_qps(all_results, str(output))
        plot_latency_cdf(all_results, slos, str(output))
        plot_fairness_heatmap(all_results, str(output))
        print(f"  Charts saved to {output}/")

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write test_main.py**

```python
import subprocess
import sys


class TestCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True, text=True, cwd=".",
        )
        assert result.returncode == 0
        assert "SLO-Aware" in result.stdout

    def test_list_policies(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--list-policies"],
            capture_output=True, text=True, cwd=".",
        )
        assert result.returncode == 0
        assert "fcfs" in result.stdout
        assert "slo_aware" in result.stdout
```

- [ ] **Step 3: Run all tests**

```bash
cd 07-slo-scheduling && python -m pytest tests/ -v
```

Expected: All tests pass (~30+ tests).

- [ ] **Step 4: Commit**

```bash
git add 07-slo-scheduling/src/main.py 07-slo-scheduling/tests/test_main.py
git commit -m "feat(slo-scheduling): add CLI entry point with --profile, --policy, --step"
```

---

### Task 8: README + Top-Level Integration

**Files:**
- Create: `07-slo-scheduling/README.md`
- Modify: `README.md` (top-level)

- [ ] **Step 1: Create 07-slo-scheduling/README.md**

Write README following portfolio pattern with sections: What It Does, Key Results (placeholder), Scheduling Policies, Workload Design, Hardware Profiles, Usage, Project Structure.

- [ ] **Step 2: Update top-level README.md**

Add P07 row to the Projects table:
```
| 07 | [SLO Scheduling](./07-slo-scheduling/) | Guarantee latency under load | Goodput: FCFS 30% vs SLO-Aware 80% at QPS=20 |
```

Update skill progression:
```
Systems Layer
  06 cost optimization -> 07 SLO scheduling
```

Update test count.

- [ ] **Step 3: Run full test suite**

```bash
cd 07-slo-scheduling && python -m pytest tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add 07-slo-scheduling/README.md README.md
git commit -m "docs(slo-scheduling): add README and update portfolio index"
```

---

### Task 9: Local Benchmark Run (M4)

**Depends on:** Tasks 1-7 complete, Ollama running with qwen3.5:4b

- [ ] **Step 1: Run local benchmark**

```bash
cd 07-slo-scheduling && python -m src.main --profile local --step all
```

- [ ] **Step 2: Review results and update README with local findings**

- [ ] **Step 3: Commit results**

```bash
git add 07-slo-scheduling/results/ 07-slo-scheduling/README.md
git commit -m "feat(slo-scheduling): local benchmark results on M4"
```

---

### Task 10: GPU Benchmark Run (RunPod L40S)

**Depends on:** Tasks 1-7 complete, RunPod account funded

- [ ] **Step 1: Create RunPod pod (L40S, runpod-torch-v280)**

- [ ] **Step 2: Install vLLM + deps, upload code**

- [ ] **Step 3: Run GPU benchmarks (FCFS, Priority, SLO-Aware)**

```bash
# Start vLLM with FCFS (default)
python -m src.main --profile gpu --policy fcfs --step benchmark

# Restart vLLM with --scheduling-policy priority
python -m src.main --profile gpu --policy priority --step benchmark

# SLO-aware uses proxy, works with any vLLM config
python -m src.main --profile gpu --policy slo_aware --step benchmark
```

- [ ] **Step 4: Download results, stop pod**

- [ ] **Step 5: Generate visualizations and update README**

```bash
cd 07-slo-scheduling && python -m src.main --profile gpu --step visualize
```

- [ ] **Step 6: Commit and push**

```bash
git add 07-slo-scheduling/results/ 07-slo-scheduling/README.md
git commit -m "feat(slo-scheduling): GPU benchmark results on L40S"
git push
```
