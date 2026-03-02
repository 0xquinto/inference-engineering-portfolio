# Portfolio Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the agentic gateway and empty speculative decoding projects with a quantization pipeline and prefix caching project, creating a portfolio that targets inference engineering roles.

**Architecture:** Three self-contained projects (01-quantization, 02-inference-benchmarks, 03-prefix-caching) each with src/, configs/, tests/, results/, Dockerfile, and README. Projects share patterns from existing 02-inference-benchmarks (dataclasses, async httpx, pytest, YAML configs) but are independent.

**Tech Stack:** Python 3.10+, vLLM, AutoGPTQ, AutoAWQ, llm-compressor, httpx, pytest, matplotlib, pandas, PyYAML, Docker

---

## Phase 0: Repository Cleanup

### Task 1: Remove empty speculative decoding project

**Files:**
- Delete: `03-speculative-decoding/` (entire directory)

**Step 1: Delete the directory**

```bash
rm -rf 03-speculative-decoding
```

**Step 2: Commit**

```bash
git add -A
git commit -m "Remove empty speculative decoding project

Scaffold had 0 implementation lines. Replacing with prefix caching project."
```

### Task 2: Archive agentic gateway

**Files:**
- Delete: `01-agentic-gateway/` (entire directory)

**Step 1: Delete the directory**

```bash
rm -rf 01-agentic-gateway
```

**Step 2: Commit**

```bash
git add -A
git commit -m "Remove agentic gateway from portfolio

Application-layer project (FastAPI + heuristic routing) doesn't demonstrate
inference engineering skills. Replacing with quantization pipeline."
```

---

## Phase 1: Project 01 — Quantization Pipeline

### Task 3: Create project scaffold and config

**Files:**
- Create: `01-quantization/configs/quantization.yaml`
- Create: `01-quantization/src/__init__.py`
- Create: `01-quantization/tests/__init__.py`
- Create: `01-quantization/tests/conftest.py`
- Create: `01-quantization/results/.gitkeep`
- Create: `01-quantization/scripts/.gitkeep`
- Create: `01-quantization/requirements.txt`

**Step 1: Create directory structure**

```bash
mkdir -p 01-quantization/{src,configs,results,tests,scripts}
touch 01-quantization/src/__init__.py
touch 01-quantization/tests/__init__.py
touch 01-quantization/results/.gitkeep
touch 01-quantization/scripts/.gitkeep
```

**Step 2: Write config**

`01-quantization/configs/quantization.yaml`:
```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct
  type: dense
  params: 7B
  base_precision: bf16

formats:
  bf16:
    description: "BF16 baseline (no quantization)"
    tool: null
  gptq_int4:
    description: "GPTQ INT4 weight-only quantization"
    tool: auto_gptq
    bits: 4
    group_size: 128
    calibration_samples: 128
  awq_int4:
    description: "AWQ INT4 activation-aware weight quantization"
    tool: autoawq
    bits: 4
    group_size: 128
  fp8:
    description: "FP8 E4M3 weights + KV cache"
    tool: llm_compressor
    target_dtype: fp8

evaluation:
  perplexity:
    dataset: wikitext
    subset: wikitext-2-raw-v1
    split: test
    max_samples: 500
  mmlu:
    num_tasks: 14
    num_few_shot: 5
    max_samples_per_task: 100

benchmark:
  engine: vllm
  port: 8010
  concurrency_levels: [1, 10, 50]
  warmup_requests: 3
  requests_per_prompt: 5
  max_tokens: 256
  temperature: 0.0
  prompts:
    - "What is the capital of France?"
    - "Explain quantum computing in simple terms."
    - "Write a Python function to check if a number is prime."
    - "Summarize the key differences between TCP and UDP."
```

**Step 3: Write requirements.txt**

`01-quantization/requirements.txt`:
```
httpx>=0.28.0
pandas>=2.2.0
matplotlib>=3.9.0
pyyaml>=6.0
tqdm>=4.67.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
# GPU-only deps (install on RunPod):
# auto-gptq>=0.7.0
# autoawq>=0.2.0
# llmcompressor>=0.5.0
# vllm>=0.16.0
# torch>=2.5.0
# transformers>=4.48.0
# datasets>=3.0.0
```

**Step 4: Write conftest.py**

`01-quantization/tests/conftest.py`:
```python
import pytest
import yaml
from pathlib import Path


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "configs" / "quantization.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_results():
    return {
        "bf16": {
            "perplexity": 6.82,
            "mmlu_accuracy": 0.714,
            "ttft_ms": {"1": 45.2, "10": 52.1, "50": 89.3},
            "throughput_tps": {"1": 83.1, "10": 412.5, "50": 1021.0},
            "vram_mb": 14200,
            "load_time_s": 12.3,
        },
        "gptq_int4": {
            "perplexity": 7.01,
            "mmlu_accuracy": 0.698,
            "ttft_ms": {"1": 32.1, "10": 38.5, "50": 62.7},
            "throughput_tps": {"1": 112.4, "10": 567.8, "50": 1456.0},
            "vram_mb": 5100,
            "load_time_s": 8.1,
        },
    }
```

**Step 5: Commit**

```bash
git add 01-quantization/
git commit -m "feat(quantization): add project scaffold and config"
```

---

### Task 4: Write quantization dataclasses and config loader

**Files:**
- Create: `01-quantization/src/config.py`
- Create: `01-quantization/tests/test_config.py`

**Step 1: Write the failing test**

`01-quantization/tests/test_config.py`:
```python
from pathlib import Path

from src.config import QuantFormat, QuantConfig, load_config


class TestQuantFormat:
    def test_from_dict_gptq(self):
        data = {
            "description": "GPTQ INT4",
            "tool": "auto_gptq",
            "bits": 4,
            "group_size": 128,
            "calibration_samples": 128,
        }
        fmt = QuantFormat.from_dict("gptq_int4", data)
        assert fmt.name == "gptq_int4"
        assert fmt.tool == "auto_gptq"
        assert fmt.bits == 4

    def test_from_dict_baseline(self):
        data = {"description": "BF16 baseline", "tool": None}
        fmt = QuantFormat.from_dict("bf16", data)
        assert fmt.is_baseline

    def test_vllm_model_id_baseline(self):
        fmt = QuantFormat.from_dict("bf16", {"description": "baseline", "tool": None})
        assert fmt.vllm_model_path("Qwen/Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-7B-Instruct"

    def test_vllm_model_id_quantized(self):
        fmt = QuantFormat.from_dict("gptq_int4", {"description": "GPTQ", "tool": "auto_gptq", "bits": 4})
        path = fmt.vllm_model_path("Qwen/Qwen2.5-7B-Instruct")
        assert path == "quantized_models/Qwen2.5-7B-Instruct-gptq_int4"


class TestLoadConfig:
    def test_loads_yaml(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "quantization.yaml")
        assert isinstance(cfg, QuantConfig)
        assert cfg.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert len(cfg.formats) == 4
        assert "bf16" in [f.name for f in cfg.formats]

    def test_benchmark_prompts(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "quantization.yaml")
        assert len(cfg.benchmark_prompts) > 0

    def test_concurrency_levels(self):
        cfg = load_config(Path(__file__).parent.parent / "configs" / "quantization.yaml")
        assert cfg.concurrency_levels == [1, 10, 50]
```

**Step 2: Run test to verify it fails**

```bash
cd 01-quantization && python -m pytest tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

**Step 3: Write implementation**

`01-quantization/src/config.py`:
```python
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class QuantFormat:
    name: str
    description: str
    tool: str | None = None
    bits: int | None = None
    group_size: int | None = None
    calibration_samples: int | None = None
    target_dtype: str | None = None

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "QuantFormat":
        return cls(
            name=name,
            description=data.get("description", ""),
            tool=data.get("tool"),
            bits=data.get("bits"),
            group_size=data.get("group_size"),
            calibration_samples=data.get("calibration_samples"),
            target_dtype=data.get("target_dtype"),
        )

    @property
    def is_baseline(self) -> bool:
        return self.tool is None

    def vllm_model_path(self, base_model: str) -> str:
        if self.is_baseline:
            return base_model
        short_name = base_model.split("/")[-1]
        return f"quantized_models/{short_name}-{self.name}"


@dataclass
class QuantConfig:
    model_name: str
    formats: list[QuantFormat]
    benchmark_prompts: list[str]
    concurrency_levels: list[int]
    engine_port: int = 8010
    warmup_requests: int = 3
    requests_per_prompt: int = 5
    max_tokens: int = 256
    temperature: float = 0.0
    perplexity_max_samples: int = 500
    mmlu_num_tasks: int = 14
    mmlu_few_shot: int = 5


def load_config(path: Path) -> QuantConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    formats = [
        QuantFormat.from_dict(name, data)
        for name, data in raw["formats"].items()
    ]

    bench = raw["benchmark"]
    eval_cfg = raw.get("evaluation", {})

    return QuantConfig(
        model_name=raw["model"]["name"],
        formats=formats,
        benchmark_prompts=bench.get("prompts", []),
        concurrency_levels=bench.get("concurrency_levels", [1, 10, 50]),
        engine_port=bench.get("port", 8010),
        warmup_requests=bench.get("warmup_requests", 3),
        requests_per_prompt=bench.get("requests_per_prompt", 5),
        max_tokens=bench.get("max_tokens", 256),
        temperature=bench.get("temperature", 0.0),
        perplexity_max_samples=eval_cfg.get("perplexity", {}).get("max_samples", 500),
        mmlu_num_tasks=eval_cfg.get("mmlu", {}).get("num_tasks", 14),
        mmlu_few_shot=eval_cfg.get("mmlu", {}).get("num_few_shot", 5),
    )
```

**Step 4: Run test to verify it passes**

```bash
cd 01-quantization && python -m pytest tests/test_config.py -v
```
Expected: 6 passed

**Step 5: Commit**

```bash
git add 01-quantization/src/config.py 01-quantization/tests/test_config.py
git commit -m "feat(quantization): add config loader and QuantFormat dataclass"
```

---

### Task 5: Write quantization runner (mocked for local testing)

**Files:**
- Create: `01-quantization/src/quantize.py`
- Create: `01-quantization/tests/test_quantize.py`

**Step 1: Write the failing test**

`01-quantization/tests/test_quantize.py`:
```python
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.config import QuantFormat
from src.quantize import QuantizationRunner, QuantizeResult


class TestQuantizeResult:
    def test_dataclass_fields(self):
        r = QuantizeResult(
            format_name="gptq_int4",
            output_path="quantized_models/Qwen-gptq_int4",
            time_seconds=120.5,
            original_size_mb=14000,
            quantized_size_mb=4200,
        )
        assert r.compression_ratio == 14000 / 4200

    def test_compression_ratio_baseline(self):
        r = QuantizeResult(
            format_name="bf16",
            output_path="Qwen/Qwen2.5-7B-Instruct",
            time_seconds=0.0,
            original_size_mb=14000,
            quantized_size_mb=14000,
        )
        assert r.compression_ratio == 1.0


class TestQuantizationRunner:
    def test_skip_baseline(self):
        fmt = QuantFormat.from_dict("bf16", {"description": "baseline", "tool": None})
        runner = QuantizationRunner(model_name="Qwen/Qwen2.5-7B-Instruct")
        result = runner.quantize(fmt)
        assert result.format_name == "bf16"
        assert result.time_seconds == 0.0

    @patch("src.quantize.QuantizationRunner._run_gptq")
    def test_dispatches_gptq(self, mock_gptq):
        mock_gptq.return_value = QuantizeResult(
            format_name="gptq_int4",
            output_path="quantized_models/test-gptq_int4",
            time_seconds=60.0,
            original_size_mb=14000,
            quantized_size_mb=4200,
        )
        fmt = QuantFormat.from_dict("gptq_int4", {
            "description": "GPTQ", "tool": "auto_gptq", "bits": 4, "group_size": 128,
        })
        runner = QuantizationRunner(model_name="test/model")
        result = runner.quantize(fmt)
        mock_gptq.assert_called_once_with(fmt)
        assert result.format_name == "gptq_int4"

    @patch("src.quantize.QuantizationRunner._run_awq")
    def test_dispatches_awq(self, mock_awq):
        mock_awq.return_value = QuantizeResult(
            format_name="awq_int4",
            output_path="quantized_models/test-awq_int4",
            time_seconds=45.0,
            original_size_mb=14000,
            quantized_size_mb=4100,
        )
        fmt = QuantFormat.from_dict("awq_int4", {
            "description": "AWQ", "tool": "autoawq", "bits": 4, "group_size": 128,
        })
        runner = QuantizationRunner(model_name="test/model")
        result = runner.quantize(fmt)
        mock_awq.assert_called_once_with(fmt)

    def test_unknown_tool_raises(self):
        fmt = QuantFormat.from_dict("unknown", {
            "description": "Unknown", "tool": "nonexistent_tool",
        })
        runner = QuantizationRunner(model_name="test/model")
        import pytest
        with pytest.raises(ValueError, match="Unsupported quantization tool"):
            runner.quantize(fmt)
```

**Step 2: Run test to verify it fails**

```bash
cd 01-quantization && python -m pytest tests/test_quantize.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`01-quantization/src/quantize.py`:
```python
import time
from dataclasses import dataclass
from pathlib import Path

from .config import QuantFormat


@dataclass
class QuantizeResult:
    format_name: str
    output_path: str
    time_seconds: float
    original_size_mb: int
    quantized_size_mb: int

    @property
    def compression_ratio(self) -> float:
        if self.quantized_size_mb == 0:
            return 0.0
        return self.original_size_mb / self.quantized_size_mb


class QuantizationRunner:
    def __init__(self, model_name: str, output_dir: str = "quantized_models"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)

    def quantize(self, fmt: QuantFormat) -> QuantizeResult:
        if fmt.is_baseline:
            return QuantizeResult(
                format_name=fmt.name,
                output_path=self.model_name,
                time_seconds=0.0,
                original_size_mb=0,
                quantized_size_mb=0,
            )

        dispatch = {
            "auto_gptq": self._run_gptq,
            "autoawq": self._run_awq,
            "llm_compressor": self._run_fp8,
        }

        if fmt.tool not in dispatch:
            raise ValueError(f"Unsupported quantization tool: {fmt.tool}")

        return dispatch[fmt.tool](fmt)

    def _output_path(self, fmt: QuantFormat) -> Path:
        short_name = self.model_name.split("/")[-1]
        return self.output_dir / f"{short_name}-{fmt.name}"

    def _run_gptq(self, fmt: QuantFormat) -> QuantizeResult:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer

        output_path = self._output_path(fmt)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        quant_config = BaseQuantizeConfig(
            bits=fmt.bits or 4,
            group_size=fmt.group_size or 128,
        )

        model = AutoGPTQForCausalLM.from_pretrained(
            self.model_name, quant_config
        )

        start = time.time()
        model.quantize(tokenizer, calibration_samples=fmt.calibration_samples or 128)
        elapsed = time.time() - start

        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        orig_size = _dir_size_mb(Path(model.config._name_or_path))
        quant_size = _dir_size_mb(output_path)

        return QuantizeResult(
            format_name=fmt.name,
            output_path=str(output_path),
            time_seconds=elapsed,
            original_size_mb=orig_size,
            quantized_size_mb=quant_size,
        )

    def _run_awq(self, fmt: QuantFormat) -> QuantizeResult:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        output_path = self._output_path(fmt)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoAWQForCausalLM.from_pretrained(self.model_name)

        quant_config = {
            "w_bit": fmt.bits or 4,
            "q_group_size": fmt.group_size or 128,
            "zero_point": True,
        }

        start = time.time()
        model.quantize(tokenizer, quant_config=quant_config)
        elapsed = time.time() - start

        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        return QuantizeResult(
            format_name=fmt.name,
            output_path=str(output_path),
            time_seconds=elapsed,
            original_size_mb=0,
            quantized_size_mb=_dir_size_mb(output_path),
        )

    def _run_fp8(self, fmt: QuantFormat) -> QuantizeResult:
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor import oneshot
        from transformers import AutoModelForCausalLM, AutoTokenizer

        output_path = self._output_path(fmt)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        recipe = QuantizationModifier(
            targets="Linear", scheme="FP8", ignore=["lm_head"]
        )

        start = time.time()
        oneshot(model=model, recipe=recipe, output_dir=str(output_path))
        elapsed = time.time() - start

        tokenizer.save_pretrained(str(output_path))

        return QuantizeResult(
            format_name=fmt.name,
            output_path=str(output_path),
            time_seconds=elapsed,
            original_size_mb=0,
            quantized_size_mb=_dir_size_mb(output_path),
        )


def _dir_size_mb(path: Path) -> int:
    if not path.exists():
        return 0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return int(total / (1024 * 1024))
```

**Step 4: Run test to verify it passes**

```bash
cd 01-quantization && python -m pytest tests/test_quantize.py -v
```
Expected: 5 passed

**Step 5: Commit**

```bash
git add 01-quantization/src/quantize.py 01-quantization/tests/test_quantize.py
git commit -m "feat(quantization): add quantization runner with GPTQ/AWQ/FP8 dispatch"
```

---

### Task 6: Write quality evaluation module

**Files:**
- Create: `01-quantization/src/evaluate.py`
- Create: `01-quantization/tests/test_evaluate.py`

**Step 1: Write the failing test**

`01-quantization/tests/test_evaluate.py`:
```python
import pytest
from unittest.mock import patch, MagicMock

from src.evaluate import EvalResult, QualityEvaluator


class TestEvalResult:
    def test_delta_from_baseline(self):
        baseline = EvalResult(format_name="bf16", perplexity=6.82, mmlu_accuracy=0.714)
        quantized = EvalResult(format_name="gptq_int4", perplexity=7.01, mmlu_accuracy=0.698)
        delta = quantized.delta_from(baseline)
        assert delta["perplexity_delta"] == pytest.approx(0.19, abs=0.01)
        assert delta["mmlu_delta"] == pytest.approx(-0.016, abs=0.001)

    def test_delta_percentage(self):
        baseline = EvalResult(format_name="bf16", perplexity=6.82, mmlu_accuracy=0.714)
        quantized = EvalResult(format_name="gptq_int4", perplexity=7.01, mmlu_accuracy=0.698)
        delta = quantized.delta_from(baseline)
        assert delta["perplexity_pct"] == pytest.approx(2.79, abs=0.1)
        assert delta["mmlu_pct"] == pytest.approx(-2.24, abs=0.1)

    def test_to_dict(self):
        r = EvalResult(format_name="bf16", perplexity=6.82, mmlu_accuracy=0.714)
        d = r.to_dict()
        assert d["format_name"] == "bf16"
        assert d["perplexity"] == 6.82


class TestQualityEvaluator:
    def test_init(self):
        evaluator = QualityEvaluator(model_path="test/model", max_samples=100)
        assert evaluator.model_path == "test/model"
        assert evaluator.max_samples == 100
```

**Step 2: Run test to verify it fails**

```bash
cd 01-quantization && python -m pytest tests/test_evaluate.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`01-quantization/src/evaluate.py`:
```python
from dataclasses import dataclass


@dataclass
class EvalResult:
    format_name: str
    perplexity: float
    mmlu_accuracy: float

    def delta_from(self, baseline: "EvalResult") -> dict:
        ppl_delta = self.perplexity - baseline.perplexity
        mmlu_delta = self.mmlu_accuracy - baseline.mmlu_accuracy
        return {
            "perplexity_delta": ppl_delta,
            "perplexity_pct": (ppl_delta / baseline.perplexity) * 100,
            "mmlu_delta": mmlu_delta,
            "mmlu_pct": (mmlu_delta / baseline.mmlu_accuracy) * 100,
        }

    def to_dict(self) -> dict:
        return {
            "format_name": self.format_name,
            "perplexity": self.perplexity,
            "mmlu_accuracy": self.mmlu_accuracy,
        }


class QualityEvaluator:
    """Evaluates model quality via perplexity and MMLU accuracy.

    GPU-only: methods call transformers + datasets at runtime.
    """

    def __init__(self, model_path: str, max_samples: int = 500):
        self.model_path = model_path
        self.max_samples = max_samples

    def compute_perplexity(self) -> float:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto"
        )
        model.eval()

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in dataset["text"] if len(t.strip()) > 0][:self.max_samples]

        encodings = tokenizer("\n\n".join(texts), return_tensors="pt", truncation=True, max_length=2048)
        input_ids = encodings.input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        return torch.exp(outputs.loss).item()

    def compute_mmlu_accuracy(self, num_tasks: int = 14, few_shot: int = 5) -> float:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto"
        )
        model.eval()

        tasks = [
            "abstract_algebra", "anatomy", "astronomy", "college_biology",
            "college_chemistry", "college_physics", "computer_science",
            "econometrics", "high_school_biology", "high_school_chemistry",
            "high_school_mathematics", "high_school_physics",
            "machine_learning", "professional_medicine",
        ][:num_tasks]

        correct, total = 0, 0
        choices = ["A", "B", "C", "D"]
        choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]

        for task in tasks:
            ds = load_dataset("cais/mmlu", task, split="test")
            for row in list(ds)[:100]:
                prompt = f"Question: {row['question']}\n"
                for i, opt in enumerate(row["choices"]):
                    prompt += f"{choices[i]}. {opt}\n"
                prompt += "Answer:"

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1]

                probs = torch.softmax(logits[choice_ids], dim=0)
                pred = probs.argmax().item()

                if pred == row["answer"]:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def evaluate(self, format_name: str) -> EvalResult:
        ppl = self.compute_perplexity()
        mmlu = self.compute_mmlu_accuracy()
        return EvalResult(format_name=format_name, perplexity=ppl, mmlu_accuracy=mmlu)
```

**Step 4: Run test to verify it passes**

```bash
cd 01-quantization && python -m pytest tests/test_evaluate.py -v
```
Expected: 4 passed

**Step 5: Commit**

```bash
git add 01-quantization/src/evaluate.py 01-quantization/tests/test_evaluate.py
git commit -m "feat(quantization): add quality evaluator with perplexity and MMLU"
```

---

### Task 7: Write performance benchmark module

**Files:**
- Create: `01-quantization/src/benchmark.py`
- Create: `01-quantization/tests/test_benchmark.py`

**Step 1: Write the failing test**

`01-quantization/tests/test_benchmark.py`:
```python
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.benchmark import BenchmarkResult, PerfBenchmarker


class TestBenchmarkResult:
    def test_tokens_per_sec(self):
        r = BenchmarkResult(
            format_name="gptq_int4",
            concurrency=10,
            ttft_ms=32.1,
            total_time_ms=500.0,
            tokens_generated=50,
            gpu_memory_mb=5100,
        )
        assert r.tokens_per_sec == 100.0

    def test_tokens_per_sec_zero_time(self):
        r = BenchmarkResult(
            format_name="gptq_int4",
            concurrency=1,
            ttft_ms=0.0,
            total_time_ms=0.0,
            tokens_generated=50,
            gpu_memory_mb=5100,
        )
        assert r.tokens_per_sec == 0.0

    def test_to_dict(self):
        r = BenchmarkResult(
            format_name="bf16",
            concurrency=1,
            ttft_ms=45.0,
            total_time_ms=1000.0,
            tokens_generated=100,
            gpu_memory_mb=14200,
        )
        d = r.to_dict()
        assert d["format_name"] == "bf16"
        assert d["tokens_per_sec"] == 100.0


class TestPerfBenchmarker:
    def test_init(self):
        b = PerfBenchmarker(port=8010, max_tokens=256)
        assert b.base_url == "http://localhost:8010"

    def test_aggregate_results(self):
        results = [
            BenchmarkResult("bf16", 1, 40.0, 1000.0, 100, 14000),
            BenchmarkResult("bf16", 1, 50.0, 1200.0, 100, 14000),
            BenchmarkResult("bf16", 1, 45.0, 1100.0, 100, 14000),
        ]
        agg = PerfBenchmarker.aggregate(results)
        assert agg["count"] == 3
        assert agg["ttft_p50"] == pytest.approx(45.0, abs=1.0)
        assert agg["throughput_p50"] == pytest.approx(100.0, abs=10.0)
```

**Step 2: Run test to verify it fails**

```bash
cd 01-quantization && python -m pytest tests/test_benchmark.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`01-quantization/src/benchmark.py`:
```python
import asyncio
import json
import time
from dataclasses import dataclass

import httpx


@dataclass
class BenchmarkResult:
    format_name: str
    concurrency: int
    ttft_ms: float
    total_time_ms: float
    tokens_generated: int
    gpu_memory_mb: int

    @property
    def tokens_per_sec(self) -> float:
        if self.total_time_ms <= 0:
            return 0.0
        return self.tokens_generated / (self.total_time_ms / 1000)

    def to_dict(self) -> dict:
        return {
            "format_name": self.format_name,
            "concurrency": self.concurrency,
            "ttft_ms": self.ttft_ms,
            "total_time_ms": self.total_time_ms,
            "tokens_generated": self.tokens_generated,
            "tokens_per_sec": self.tokens_per_sec,
            "gpu_memory_mb": self.gpu_memory_mb,
        }

    @staticmethod
    def aggregate(results: list["BenchmarkResult"]) -> dict:
        if not results:
            return {"count": 0}
        ttfts = sorted(r.ttft_ms for r in results)
        tps_list = sorted(r.tokens_per_sec for r in results)
        n = len(results)
        return {
            "count": n,
            "ttft_p50": ttfts[n // 2],
            "ttft_p95": ttfts[int(n * 0.95)] if n >= 20 else ttfts[-1],
            "throughput_p50": tps_list[n // 2],
            "avg_gpu_memory_mb": sum(r.gpu_memory_mb for r in results) // n,
        }


class PerfBenchmarker:
    def __init__(self, port: int = 8010, max_tokens: int = 256, temperature: float = 0.0):
        self.base_url = f"http://localhost:{port}"
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def send_request(self, prompt: str, format_name: str, concurrency: int) -> BenchmarkResult:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        ttft_ms = 0.0
        tokens = 0
        start = time.perf_counter()

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        if ttft_ms == 0.0:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        tokens += 1

        total_ms = (time.perf_counter() - start) * 1000
        gpu_mem = self._get_gpu_memory()

        return BenchmarkResult(
            format_name=format_name,
            concurrency=concurrency,
            ttft_ms=ttft_ms,
            total_time_ms=total_ms,
            tokens_generated=tokens,
            gpu_memory_mb=gpu_mem,
        )

    async def run_concurrent(
        self, prompts: list[str], concurrency: int, format_name: str
    ) -> list[BenchmarkResult]:
        sem = asyncio.Semaphore(concurrency)

        async def limited(prompt: str) -> BenchmarkResult:
            async with sem:
                return await self.send_request(prompt, format_name, concurrency)

        tasks = [limited(p) for p in prompts]
        return list(await asyncio.gather(*tasks))

    @staticmethod
    def _get_gpu_memory() -> int:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return int(result.stdout.strip().split("\n")[0])
        except Exception:
            return 0

    @staticmethod
    def aggregate(results: list[BenchmarkResult]) -> dict:
        return BenchmarkResult.aggregate(results)
```

**Step 4: Run test to verify it passes**

```bash
cd 01-quantization && python -m pytest tests/test_benchmark.py -v
```
Expected: 5 passed

**Step 5: Commit**

```bash
git add 01-quantization/src/benchmark.py 01-quantization/tests/test_benchmark.py
git commit -m "feat(quantization): add performance benchmarker with streaming metrics"
```

---

### Task 8: Write visualization module

**Files:**
- Create: `01-quantization/src/visualize.py`
- Create: `01-quantization/tests/test_visualize.py`

**Step 1: Write the failing test**

`01-quantization/tests/test_visualize.py`:
```python
import pytest

from src.visualize import prepare_comparison_data, prepare_pareto_data


class TestPrepareComparisonData:
    def test_basic(self, sample_results):
        df = prepare_comparison_data(sample_results)
        assert len(df) == 2
        assert "format" in df.columns
        assert "vram_mb" in df.columns

    def test_columns(self, sample_results):
        df = prepare_comparison_data(sample_results)
        expected_cols = {"format", "perplexity", "mmlu_accuracy", "vram_mb", "throughput_c1", "ttft_c1"}
        assert expected_cols.issubset(set(df.columns))


class TestPreparePareto:
    def test_pareto_points(self, sample_results):
        points = prepare_pareto_data(sample_results)
        assert len(points) == 2
        assert all("format" in p for p in points)
        assert all("quality" in p for p in points)
        assert all("speed" in p for p in points)
        assert all("memory" in p for p in points)
```

**Step 2: Run test to verify it fails**

```bash
cd 01-quantization && python -m pytest tests/test_visualize.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`01-quantization/src/visualize.py`:
```python
from pathlib import Path

import pandas as pd


def prepare_comparison_data(results: dict) -> pd.DataFrame:
    rows = []
    for fmt_name, data in results.items():
        row = {
            "format": fmt_name,
            "perplexity": data.get("perplexity", 0),
            "mmlu_accuracy": data.get("mmlu_accuracy", 0),
            "vram_mb": data.get("vram_mb", 0),
            "load_time_s": data.get("load_time_s", 0),
        }
        for level in ["1", "10", "50"]:
            row[f"throughput_c{level}"] = data.get("throughput_tps", {}).get(level, 0)
            row[f"ttft_c{level}"] = data.get("ttft_ms", {}).get(level, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_pareto_data(results: dict) -> list[dict]:
    points = []
    for fmt_name, data in results.items():
        points.append({
            "format": fmt_name,
            "quality": data.get("mmlu_accuracy", 0),
            "speed": data.get("throughput_tps", {}).get("1", 0),
            "memory": data.get("vram_mb", 0),
        })
    return points


def plot_comparison(results: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt

    df = prepare_comparison_data(results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Quantization Format Comparison", fontsize=14)

    axes[0, 0].barh(df["format"], df["vram_mb"])
    axes[0, 0].set_xlabel("VRAM (MB)")
    axes[0, 0].set_title("Memory Usage")

    axes[0, 1].barh(df["format"], df["perplexity"])
    axes[0, 1].set_xlabel("Perplexity (lower = better)")
    axes[0, 1].set_title("Quality: Perplexity")

    axes[1, 0].barh(df["format"], df["throughput_c1"])
    axes[1, 0].set_xlabel("Tokens/sec (concurrency=1)")
    axes[1, 0].set_title("Throughput")

    axes[1, 1].barh(df["format"], df["ttft_c1"])
    axes[1, 1].set_xlabel("TTFT (ms)")
    axes[1, 1].set_title("Time to First Token")

    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=150)
    plt.close()


def plot_pareto(results: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt

    points = prepare_pareto_data(results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    for p in points:
        ax.scatter(p["speed"], p["quality"], s=p["memory"] / 20, alpha=0.7)
        ax.annotate(p["format"], (p["speed"], p["quality"]),
                    textcoords="offset points", xytext=(10, 5))

    ax.set_xlabel("Throughput (tokens/sec, concurrency=1)")
    ax.set_ylabel("MMLU Accuracy")
    ax.set_title("Quality vs Speed (bubble size = VRAM)")
    plt.tight_layout()
    plt.savefig(out / "pareto.png", dpi=150)
    plt.close()
```

**Step 4: Run test to verify it passes**

```bash
cd 01-quantization && python -m pytest tests/test_visualize.py -v
```
Expected: 4 passed

**Step 5: Commit**

```bash
git add 01-quantization/src/visualize.py 01-quantization/tests/test_visualize.py
git commit -m "feat(quantization): add visualization with comparison charts and Pareto frontier"
```

---

### Task 9: Write CLI entrypoint

**Files:**
- Create: `01-quantization/src/main.py`
- Create: `01-quantization/tests/test_main.py`

**Step 1: Write the failing test**

`01-quantization/tests/test_main.py`:
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
        assert "Quantization Pipeline" in result.stdout

    def test_list_formats(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--list-formats"],
            capture_output=True, text=True, cwd=".",
        )
        assert result.returncode == 0
        assert "bf16" in result.stdout
        assert "gptq_int4" in result.stdout
```

**Step 2: Run test to verify it fails**

```bash
cd 01-quantization && python -m pytest tests/test_main.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`01-quantization/src/main.py`:
```python
import argparse
import asyncio
import json
from pathlib import Path

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Quantization Pipeline & Quality-Speed Tradeoffs")
    parser.add_argument("--config", default="configs/quantization.yaml", help="Config file path")
    parser.add_argument("--output", default="results/", help="Output directory for results")
    parser.add_argument("--list-formats", action="store_true", help="List available quantization formats")
    parser.add_argument("--step", choices=["quantize", "evaluate", "benchmark", "visualize", "all"],
                        default="all", help="Which pipeline step to run")
    parser.add_argument("--format", help="Run only this format (default: all)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    if args.list_formats:
        print("Available quantization formats:")
        for fmt in cfg.formats:
            baseline = " (baseline)" if fmt.is_baseline else ""
            print(f"  {fmt.name}: {fmt.description}{baseline}")
        return

    asyncio.run(async_main(args, cfg))


async def async_main(args, cfg):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    formats = cfg.formats
    if args.format:
        formats = [f for f in formats if f.name == args.format]
        if not formats:
            print(f"Error: format '{args.format}' not found")
            return

    all_results = {}

    if args.step in ("quantize", "all"):
        print("=== Step 1: Quantization ===")
        from .quantize import QuantizationRunner
        runner = QuantizationRunner(model_name=cfg.model_name)
        for fmt in formats:
            if fmt.is_baseline:
                print(f"  Skipping {fmt.name} (baseline)")
                continue
            print(f"  Quantizing {fmt.name}...")
            result = runner.quantize(fmt)
            print(f"  Done in {result.time_seconds:.1f}s, compression {result.compression_ratio:.1f}x")

    if args.step in ("evaluate", "all"):
        print("\n=== Step 2: Quality Evaluation ===")
        from .evaluate import QualityEvaluator
        for fmt in formats:
            model_path = fmt.vllm_model_path(cfg.model_name)
            print(f"  Evaluating {fmt.name} ({model_path})...")
            evaluator = QualityEvaluator(model_path=model_path, max_samples=cfg.perplexity_max_samples)
            eval_result = evaluator.evaluate(fmt.name)
            all_results.setdefault(fmt.name, {}).update(eval_result.to_dict())
            print(f"  Perplexity: {eval_result.perplexity:.2f}, MMLU: {eval_result.mmlu_accuracy:.3f}")

    if args.step in ("benchmark", "all"):
        print("\n=== Step 3: Performance Benchmarks ===")
        from .benchmark import PerfBenchmarker
        benchmarker = PerfBenchmarker(port=cfg.engine_port, max_tokens=cfg.max_tokens)
        for fmt in formats:
            print(f"  Benchmarking {fmt.name}...")
            for conc in cfg.concurrency_levels:
                prompts = cfg.benchmark_prompts * cfg.requests_per_prompt
                results = await benchmarker.run_concurrent(prompts, conc, fmt.name)
                agg = benchmarker.aggregate(results)
                fmt_data = all_results.setdefault(fmt.name, {})
                fmt_data.setdefault("ttft_ms", {})[str(conc)] = agg.get("ttft_p50", 0)
                fmt_data.setdefault("throughput_tps", {})[str(conc)] = agg.get("throughput_p50", 0)
                fmt_data["vram_mb"] = agg.get("avg_gpu_memory_mb", 0)
                print(f"    c={conc}: TTFT={agg.get('ttft_p50', 0):.1f}ms, TPS={agg.get('throughput_p50', 0):.1f}")

    if args.step in ("visualize", "all"):
        print("\n=== Step 4: Visualization ===")
        from .visualize import plot_comparison, plot_pareto
        plot_comparison(all_results, str(output))
        plot_pareto(all_results, str(output))
        print(f"  Charts saved to {output}/")

    results_file = output / "quantization_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

```bash
cd 01-quantization && python -m pytest tests/test_main.py -v
```
Expected: 2 passed

**Step 5: Commit**

```bash
git add 01-quantization/src/main.py 01-quantization/tests/test_main.py
git commit -m "feat(quantization): add CLI entrypoint with step-by-step pipeline"
```

---

### Task 10: Write GPU setup and run scripts

**Files:**
- Create: `01-quantization/scripts/setup_gpu.sh`
- Create: `01-quantization/scripts/run_all.sh`

**Step 1: Write setup script**

`01-quantization/scripts/setup_gpu.sh`:
```bash
#!/bin/bash
set -e

echo "=== Setting up quantization project on GPU instance ==="

# Link HF cache if on RunPod
if [ -d "/workspace" ]; then
    mkdir -p /workspace/hf_cache
    ln -sf /workspace/hf_cache /root/.cache/huggingface
    echo "Linked HF cache to /workspace/hf_cache"
fi

# Install Python dependencies
pip install --upgrade pip
pip install auto-gptq autoawq llmcompressor
pip install vllm>=0.16.0
pip install httpx pandas matplotlib pyyaml tqdm
pip install transformers datasets torch
pip install pytest pytest-asyncio

# Download model
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Qwen2.5-7B-Instruct...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype='auto')
print('Done.')
"

echo "=== Setup complete ==="
```

**Step 2: Write run script**

`01-quantization/scripts/run_all.sh`:
```bash
#!/bin/bash
set -e

echo "=== Running full quantization pipeline ==="

cd "$(dirname "$0")/.."

# Step 1: Quantize
python -m src.main --step quantize
echo ""

# Step 2: Evaluate quality (runs each model on GPU)
python -m src.main --step evaluate
echo ""

# Step 3: Benchmark each format via vLLM (starts/stops server per format)
# Note: this step requires manually starting vLLM with each model.
# See README for per-format benchmark instructions.
echo "Step 3 (benchmark) requires starting vLLM per format."
echo "Run: python -m src.main --step benchmark --format <format_name>"
echo ""

# Step 4: Visualize
python -m src.main --step visualize
echo ""

echo "=== Pipeline complete. Results in results/ ==="
```

**Step 3: Make executable**

```bash
chmod +x 01-quantization/scripts/setup_gpu.sh 01-quantization/scripts/run_all.sh
```

**Step 4: Commit**

```bash
git add 01-quantization/scripts/
git commit -m "feat(quantization): add GPU setup and run scripts"
```

---

## Phase 2: Project 03 — Prefix Caching

### Task 11: Create project scaffold and config

**Files:**
- Create: `03-prefix-caching/configs/scenarios.yaml`
- Create: `03-prefix-caching/configs/engines.yaml`
- Create: `03-prefix-caching/src/__init__.py`
- Create: `03-prefix-caching/tests/__init__.py`
- Create: `03-prefix-caching/tests/conftest.py`
- Create: `03-prefix-caching/results/.gitkeep`
- Create: `03-prefix-caching/scripts/.gitkeep`
- Create: `03-prefix-caching/requirements.txt`

**Step 1: Create directory structure**

```bash
mkdir -p 03-prefix-caching/{src,configs,results,tests,scripts}
touch 03-prefix-caching/src/__init__.py
touch 03-prefix-caching/tests/__init__.py
touch 03-prefix-caching/results/.gitkeep
touch 03-prefix-caching/scripts/.gitkeep
```

**Step 2: Write scenario config**

`03-prefix-caching/configs/scenarios.yaml`:
```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct
  type: dense
  params: 7B

scenarios:
  shared_system_prompt:
    description: "100 requests with identical system prompt, varying user messages"
    system_prompt_tokens: 1000
    num_requests: 100
    user_message_tokens: 50

  multi_turn:
    description: "10-turn conversations with growing history"
    num_conversations: 10
    turns_per_conversation: 10
    tokens_per_turn: 100

  rag_common_context:
    description: "50 questions against the same retrieved document"
    context_tokens: 3000
    num_queries: 50
    query_tokens: 30

  cache_pressure:
    description: "Increasing unique prefixes to find eviction cliff"
    prefix_tokens: 500
    unique_prefixes: [10, 25, 50, 100, 200, 500]
    requests_per_prefix: 5

benchmark:
  concurrency: 10
  warmup_requests: 3
  max_tokens: 128
  temperature: 0.0
```

**Step 3: Write engines config**

`03-prefix-caching/configs/engines.yaml`:
```yaml
engines:
  vllm:
    port: 8010
    prefix_caching_flag: "--enable-prefix-caching"
    extra_args:
      - "--max-model-len"
      - "8192"

  sglang:
    port: 8020
    prefix_caching_flag: null  # RadixAttention is always on; disable with --disable-radix-cache
    extra_args:
      - "--context-length"
      - "8192"
```

**Step 4: Write requirements.txt**

`03-prefix-caching/requirements.txt`:
```
httpx>=0.28.0
pandas>=2.2.0
matplotlib>=3.9.0
pyyaml>=6.0
tqdm>=4.67.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
# GPU-only:
# vllm>=0.16.0
# sglang[all]>=0.5.0
```

**Step 5: Write conftest.py**

`03-prefix-caching/tests/conftest.py`:
```python
import pytest
import yaml
from pathlib import Path


@pytest.fixture
def scenario_config():
    path = Path(__file__).parent.parent / "configs" / "scenarios.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def engines_config():
    path = Path(__file__).parent.parent / "configs" / "engines.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
```

**Step 6: Commit**

```bash
git add 03-prefix-caching/
git commit -m "feat(prefix-caching): add project scaffold and configs"
```

---

### Task 12: Write scenario prompt generators

**Files:**
- Create: `03-prefix-caching/src/scenarios.py`
- Create: `03-prefix-caching/tests/test_scenarios.py`

**Step 1: Write the failing test**

`03-prefix-caching/tests/test_scenarios.py`:
```python
import pytest

from src.scenarios import (
    generate_shared_system_prompt,
    generate_multi_turn,
    generate_rag_context,
    generate_cache_pressure,
)


class TestSharedSystemPrompt:
    def test_all_share_same_system(self):
        requests = generate_shared_system_prompt(num_requests=10, system_tokens=100, user_tokens=20)
        system_msgs = [r[0] for r in requests]
        assert all(s == system_msgs[0] for s in system_msgs)
        assert len(requests) == 10

    def test_user_messages_vary(self):
        requests = generate_shared_system_prompt(num_requests=10, system_tokens=100, user_tokens=20)
        user_msgs = [r[1] for r in requests]
        assert len(set(m["content"] for m in user_msgs)) > 1

    def test_message_format(self):
        requests = generate_shared_system_prompt(num_requests=1, system_tokens=50, user_tokens=20)
        system, user = requests[0]
        assert system["role"] == "system"
        assert user["role"] == "user"


class TestMultiTurn:
    def test_turn_count(self):
        convos = generate_multi_turn(num_conversations=2, turns=5, tokens_per_turn=30)
        assert len(convos) == 2
        assert len(convos[0]) == 5

    def test_growing_history(self):
        convos = generate_multi_turn(num_conversations=1, turns=5, tokens_per_turn=30)
        for i, turn_messages in enumerate(convos[0]):
            # Each turn has all previous messages + new user message
            user_msgs = [m for m in turn_messages if m["role"] == "user"]
            assert len(user_msgs) == i + 1


class TestRAGContext:
    def test_shared_context(self):
        requests = generate_rag_context(context_tokens=200, num_queries=5, query_tokens=20)
        contexts = [r[0]["content"] for r in requests]
        assert all(c == contexts[0] for c in contexts)
        assert len(requests) == 5

    def test_different_queries(self):
        requests = generate_rag_context(context_tokens=200, num_queries=5, query_tokens=20)
        queries = [r[1]["content"] for r in requests]
        assert len(set(queries)) > 1


class TestCachePressure:
    def test_unique_prefix_count(self):
        batches = generate_cache_pressure(
            prefix_tokens=50, unique_prefixes=[5, 10], requests_per_prefix=3
        )
        assert len(batches) == 2
        assert len(batches[0]) == 15  # 5 prefixes * 3 requests
        assert len(batches[1]) == 30  # 10 prefixes * 3 requests
```

**Step 2: Run test to verify it fails**

```bash
cd 03-prefix-caching && python -m pytest tests/test_scenarios.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`03-prefix-caching/src/scenarios.py`:
```python
import hashlib


# Deterministic filler text generation (no external deps)
_FILLER_WORDS = (
    "the quick brown fox jumps over the lazy dog while the cat sleeps "
    "on a warm sunny afternoon near the old oak tree by the river bank "
    "where fish swim upstream against the gentle current of clear water "
    "flowing down from the snow capped mountains in the distance beyond "
)


def _filler_text(seed: int, approx_tokens: int) -> str:
    """Generate deterministic filler text of approximately `approx_tokens` words."""
    words = _FILLER_WORDS.split()
    h = hashlib.md5(str(seed).encode()).hexdigest()
    offset = int(h[:8], 16) % len(words)
    result = []
    for i in range(approx_tokens):
        result.append(words[(offset + i) % len(words)])
    return " ".join(result)


def generate_shared_system_prompt(
    num_requests: int, system_tokens: int, user_tokens: int
) -> list[tuple[dict, dict]]:
    system_msg = {
        "role": "system",
        "content": f"You are a helpful assistant. {_filler_text(seed=0, approx_tokens=system_tokens)}",
    }
    requests = []
    for i in range(num_requests):
        user_msg = {
            "role": "user",
            "content": f"Question {i}: {_filler_text(seed=1000 + i, approx_tokens=user_tokens)}",
        }
        requests.append((system_msg, user_msg))
    return requests


def generate_multi_turn(
    num_conversations: int, turns: int, tokens_per_turn: int
) -> list[list[list[dict]]]:
    conversations = []
    for c in range(num_conversations):
        history = []
        turn_snapshots = []
        for t in range(turns):
            user_msg = {
                "role": "user",
                "content": f"Turn {t}: {_filler_text(seed=2000 + c * 100 + t, approx_tokens=tokens_per_turn)}",
            }
            history.append(user_msg)
            if t < turns - 1:
                assistant_msg = {
                    "role": "assistant",
                    "content": f"Response to turn {t}.",
                }
                history.append(assistant_msg)
            turn_snapshots.append(list(history))
        conversations.append(turn_snapshots)
    return conversations


def generate_rag_context(
    context_tokens: int, num_queries: int, query_tokens: int
) -> list[tuple[dict, dict]]:
    context_msg = {
        "role": "system",
        "content": (
            f"Use the following context to answer the user's question.\n\n"
            f"Context: {_filler_text(seed=3000, approx_tokens=context_tokens)}"
        ),
    }
    requests = []
    for i in range(num_queries):
        query_msg = {
            "role": "user",
            "content": f"Query {i}: {_filler_text(seed=4000 + i, approx_tokens=query_tokens)}",
        }
        requests.append((context_msg, query_msg))
    return requests


def generate_cache_pressure(
    prefix_tokens: int, unique_prefixes: list[int], requests_per_prefix: int
) -> list[list[tuple[dict, dict]]]:
    batches = []
    for n_prefixes in unique_prefixes:
        batch = []
        for p in range(n_prefixes):
            system_msg = {
                "role": "system",
                "content": f"Prefix {p}: {_filler_text(seed=5000 + p, approx_tokens=prefix_tokens)}",
            }
            for r in range(requests_per_prefix):
                user_msg = {
                    "role": "user",
                    "content": f"Request {r} for prefix {p}.",
                }
                batch.append((system_msg, user_msg))
        batches.append(batch)
    return batches
```

**Step 4: Run test to verify it passes**

```bash
cd 03-prefix-caching && python -m pytest tests/test_scenarios.py -v
```
Expected: 8 passed

**Step 5: Commit**

```bash
git add 03-prefix-caching/src/scenarios.py 03-prefix-caching/tests/test_scenarios.py
git commit -m "feat(prefix-caching): add scenario prompt generators for 4 benchmark scenarios"
```

---

### Task 13: Write metrics module

**Files:**
- Create: `03-prefix-caching/src/metrics.py`
- Create: `03-prefix-caching/tests/test_metrics.py`

**Step 1: Write the failing test**

`03-prefix-caching/tests/test_metrics.py`:
```python
import pytest

from src.metrics import CacheMetrics, RequestMetric


class TestRequestMetric:
    def test_to_dict(self):
        m = RequestMetric(scenario="shared_system", caching=True, request_idx=0, ttft_ms=12.3, total_ms=500.0, tokens=50)
        d = m.to_dict()
        assert d["ttft_ms"] == 12.3
        assert d["tokens_per_sec"] == 100.0

    def test_tps_zero_time(self):
        m = RequestMetric(scenario="test", caching=True, request_idx=0, ttft_ms=0, total_ms=0, tokens=50)
        assert m.tokens_per_sec == 0.0


class TestCacheMetrics:
    def test_record_and_summary(self):
        cm = CacheMetrics()
        cm.record(RequestMetric("s1", True, 0, 10.0, 500.0, 50))
        cm.record(RequestMetric("s1", True, 1, 12.0, 520.0, 50))
        cm.record(RequestMetric("s1", True, 2, 11.0, 510.0, 50))

        summary = cm.summary("s1", caching=True)
        assert summary["count"] == 3
        assert summary["ttft_p50"] == pytest.approx(11.0, abs=1.0)

    def test_separate_caching_on_off(self):
        cm = CacheMetrics()
        cm.record(RequestMetric("s1", True, 0, 10.0, 500.0, 50))
        cm.record(RequestMetric("s1", False, 0, 40.0, 600.0, 50))

        on = cm.summary("s1", caching=True)
        off = cm.summary("s1", caching=False)
        assert on["ttft_p50"] < off["ttft_p50"]

    def test_speedup(self):
        cm = CacheMetrics()
        for i in range(5):
            cm.record(RequestMetric("s1", False, i, 40.0, 600.0, 50))
            cm.record(RequestMetric("s1", True, i, 10.0, 500.0, 50))

        speedup = cm.ttft_speedup("s1")
        assert speedup == pytest.approx(4.0, abs=0.5)

    def test_empty_summary(self):
        cm = CacheMetrics()
        summary = cm.summary("nonexistent", caching=True)
        assert summary["count"] == 0
```

**Step 2: Run test to verify it fails**

```bash
cd 03-prefix-caching && python -m pytest tests/test_metrics.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`03-prefix-caching/src/metrics.py`:
```python
from dataclasses import dataclass, field


@dataclass
class RequestMetric:
    scenario: str
    caching: bool
    request_idx: int
    ttft_ms: float
    total_ms: float
    tokens: int

    @property
    def tokens_per_sec(self) -> float:
        if self.total_ms <= 0:
            return 0.0
        return self.tokens / (self.total_ms / 1000)

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "caching": self.caching,
            "request_idx": self.request_idx,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "tokens": self.tokens,
            "tokens_per_sec": self.tokens_per_sec,
        }


class CacheMetrics:
    def __init__(self):
        self._records: list[RequestMetric] = []

    def record(self, metric: RequestMetric) -> None:
        self._records.append(metric)

    def _filter(self, scenario: str, caching: bool) -> list[RequestMetric]:
        return [r for r in self._records if r.scenario == scenario and r.caching == caching]

    def summary(self, scenario: str, caching: bool) -> dict:
        records = self._filter(scenario, caching)
        if not records:
            return {"count": 0}

        ttfts = sorted(r.ttft_ms for r in records)
        tps_list = sorted(r.tokens_per_sec for r in records)
        n = len(records)

        return {
            "count": n,
            "caching": caching,
            "ttft_p50": ttfts[n // 2],
            "ttft_p95": ttfts[int(n * 0.95)] if n >= 20 else ttfts[-1],
            "ttft_mean": sum(ttfts) / n,
            "throughput_p50": tps_list[n // 2],
        }

    def ttft_speedup(self, scenario: str) -> float:
        off = self.summary(scenario, caching=False)
        on = self.summary(scenario, caching=True)
        if on["count"] == 0 or off["count"] == 0 or on["ttft_p50"] == 0:
            return 0.0
        return off["ttft_p50"] / on["ttft_p50"]
```

**Step 4: Run test to verify it passes**

```bash
cd 03-prefix-caching && python -m pytest tests/test_metrics.py -v
```
Expected: 5 passed

**Step 5: Commit**

```bash
git add 03-prefix-caching/src/metrics.py 03-prefix-caching/tests/test_metrics.py
git commit -m "feat(prefix-caching): add CacheMetrics tracker with speedup calculation"
```

---

### Task 14: Write benchmark runner

**Files:**
- Create: `03-prefix-caching/src/benchmark.py`
- Create: `03-prefix-caching/tests/test_benchmark.py`

**Step 1: Write the failing test**

`03-prefix-caching/tests/test_benchmark.py`:
```python
import pytest

from src.benchmark import CacheBenchmarker


class TestCacheBenchmarker:
    def test_init(self):
        b = CacheBenchmarker(port=8010, max_tokens=128)
        assert b.base_url == "http://localhost:8010"

    def test_build_payload_single_turn(self):
        b = CacheBenchmarker(port=8010)
        system = {"role": "system", "content": "You are helpful."}
        user = {"role": "user", "content": "Hello"}
        payload = b._build_payload([system, user])
        assert payload["messages"] == [system, user]
        assert payload["stream"] is True

    def test_build_payload_multi_turn(self):
        b = CacheBenchmarker(port=8010)
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ]
        payload = b._build_payload(messages)
        assert len(payload["messages"]) == 3
```

**Step 2: Run test to verify it fails**

```bash
cd 03-prefix-caching && python -m pytest tests/test_benchmark.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`03-prefix-caching/src/benchmark.py`:
```python
import asyncio
import json
import time

import httpx

from .metrics import RequestMetric, CacheMetrics


class CacheBenchmarker:
    def __init__(self, port: int = 8010, max_tokens: int = 128, temperature: float = 0.0):
        self.base_url = f"http://localhost:{port}"
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _build_payload(self, messages: list[dict]) -> dict:
        return {
            "model": "default",
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

    async def send_request(
        self, messages: list[dict], scenario: str, caching: bool, request_idx: int
    ) -> RequestMetric:
        payload = self._build_payload(messages)

        ttft_ms = 0.0
        tokens = 0
        start = time.perf_counter()

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        if ttft_ms == 0.0:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        tokens += 1

        total_ms = (time.perf_counter() - start) * 1000

        return RequestMetric(
            scenario=scenario,
            caching=caching,
            request_idx=request_idx,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens=tokens,
        )

    async def run_scenario_shared_system(
        self, requests: list[tuple[dict, dict]], scenario: str, caching: bool, concurrency: int
    ) -> list[RequestMetric]:
        sem = asyncio.Semaphore(concurrency)

        async def limited(idx: int, system: dict, user: dict) -> RequestMetric:
            async with sem:
                return await self.send_request([system, user], scenario, caching, idx)

        tasks = [limited(i, sys, usr) for i, (sys, usr) in enumerate(requests)]
        return list(await asyncio.gather(*tasks))

    async def run_scenario_multi_turn(
        self, conversations: list[list[list[dict]]], scenario: str, caching: bool
    ) -> list[RequestMetric]:
        results = []
        for conv_turns in conversations:
            for idx, messages in enumerate(conv_turns):
                metric = await self.send_request(messages, scenario, caching, idx)
                results.append(metric)
        return results

    async def run_scenario_cache_pressure(
        self, batches: list[list[tuple[dict, dict]]], scenario: str, caching: bool, concurrency: int
    ) -> dict[int, list[RequestMetric]]:
        results = {}
        for batch in batches:
            n_prefixes = len(set(sys["content"] for sys, _ in batch))
            metrics = await self.run_scenario_shared_system(batch, scenario, caching, concurrency)
            results[n_prefixes] = metrics
        return results
```

**Step 4: Run test to verify it passes**

```bash
cd 03-prefix-caching && python -m pytest tests/test_benchmark.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add 03-prefix-caching/src/benchmark.py 03-prefix-caching/tests/test_benchmark.py
git commit -m "feat(prefix-caching): add benchmark runner with scenario-specific methods"
```

---

### Task 15: Write visualization module

**Files:**
- Create: `03-prefix-caching/src/visualize.py`
- Create: `03-prefix-caching/tests/test_visualize.py`

**Step 1: Write the failing test**

`03-prefix-caching/tests/test_visualize.py`:
```python
import pytest

from src.visualize import prepare_ttft_comparison, prepare_cache_pressure_curve


class TestPrepareTTFTComparison:
    def test_basic(self):
        data = {
            "shared_system_prompt": {
                "caching_on": {"ttft_p50": 8.0, "count": 100},
                "caching_off": {"ttft_p50": 35.0, "count": 100},
            }
        }
        df = prepare_ttft_comparison(data)
        assert len(df) == 1
        assert df.iloc[0]["speedup"] == pytest.approx(4.375, abs=0.1)


class TestPrepareCachePressureCurve:
    def test_basic(self):
        data = {
            10: {"ttft_p50": 8.0},
            50: {"ttft_p50": 12.0},
            200: {"ttft_p50": 30.0},
        }
        points = prepare_cache_pressure_curve(data)
        assert len(points) == 3
        assert points[0]["unique_prefixes"] == 10
        assert points[2]["ttft_p50"] == 30.0
```

**Step 2: Run test to verify it fails**

```bash
cd 03-prefix-caching && python -m pytest tests/test_visualize.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`03-prefix-caching/src/visualize.py`:
```python
from pathlib import Path

import pandas as pd


def prepare_ttft_comparison(scenario_summaries: dict) -> pd.DataFrame:
    rows = []
    for scenario, data in scenario_summaries.items():
        on = data.get("caching_on", {})
        off = data.get("caching_off", {})
        ttft_on = on.get("ttft_p50", 0)
        ttft_off = off.get("ttft_p50", 0)
        speedup = ttft_off / ttft_on if ttft_on > 0 else 0
        rows.append({
            "scenario": scenario,
            "ttft_caching_on": ttft_on,
            "ttft_caching_off": ttft_off,
            "speedup": speedup,
        })
    return pd.DataFrame(rows)


def prepare_cache_pressure_curve(pressure_data: dict) -> list[dict]:
    points = []
    for n_prefixes, summary in sorted(pressure_data.items()):
        points.append({
            "unique_prefixes": n_prefixes,
            "ttft_p50": summary.get("ttft_p50", 0),
        })
    return points


def plot_ttft_comparison(scenario_summaries: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt

    df = prepare_ttft_comparison(scenario_summaries)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = range(len(df))
    width = 0.35
    ax1.bar([i - width/2 for i in x], df["ttft_caching_off"], width, label="Caching OFF", color="#e74c3c")
    ax1.bar([i + width/2 for i in x], df["ttft_caching_on"], width, label="Caching ON", color="#2ecc71")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["scenario"], rotation=15, ha="right")
    ax1.set_ylabel("TTFT p50 (ms)")
    ax1.set_title("Time to First Token: Caching ON vs OFF")
    ax1.legend()

    ax2.barh(df["scenario"], df["speedup"], color="#3498db")
    ax2.set_xlabel("TTFT Speedup (x)")
    ax2.set_title("Prefix Caching Speedup by Scenario")

    plt.tight_layout()
    plt.savefig(out / "ttft_comparison.png", dpi=150)
    plt.close()


def plot_cache_pressure(pressure_data: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt

    points = prepare_cache_pressure_curve(pressure_data)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = [p["unique_prefixes"] for p in points]
    y = [p["ttft_p50"] for p in points]
    ax.plot(x, y, marker="o", linewidth=2, color="#e74c3c")
    ax.set_xlabel("Number of Unique Prefixes in Cache")
    ax.set_ylabel("TTFT p50 (ms)")
    ax.set_title("Cache Pressure: TTFT vs Unique Prefix Count")
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(out / "cache_pressure.png", dpi=150)
    plt.close()
```

**Step 4: Run test to verify it passes**

```bash
cd 03-prefix-caching && python -m pytest tests/test_visualize.py -v
```
Expected: 2 passed

**Step 5: Commit**

```bash
git add 03-prefix-caching/src/visualize.py 03-prefix-caching/tests/test_visualize.py
git commit -m "feat(prefix-caching): add visualization with TTFT comparison and cache pressure charts"
```

---

### Task 16: Write CLI entrypoint

**Files:**
- Create: `03-prefix-caching/src/main.py`
- Create: `03-prefix-caching/tests/test_main.py`

**Step 1: Write the failing test**

`03-prefix-caching/tests/test_main.py`:
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
        assert "Prefix Caching" in result.stdout

    def test_list_scenarios(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--list-scenarios"],
            capture_output=True, text=True, cwd=".",
        )
        assert result.returncode == 0
        assert "shared_system_prompt" in result.stdout
        assert "cache_pressure" in result.stdout
```

**Step 2: Run test to verify it fails**

```bash
cd 03-prefix-caching && python -m pytest tests/test_main.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`03-prefix-caching/src/main.py`:
```python
import argparse
import asyncio
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Prefix Caching & KV Cache Optimization Benchmarks")
    parser.add_argument("--config", default="configs/scenarios.yaml", help="Scenario config")
    parser.add_argument("--engines", default="configs/engines.yaml", help="Engine config")
    parser.add_argument("--output", default="results/", help="Output directory")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    parser.add_argument("--scenario", help="Run only this scenario")
    parser.add_argument("--engine", choices=["vllm", "sglang"], default="vllm", help="Engine to benchmark")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.engines) as f:
        engines = yaml.safe_load(f)

    if args.list_scenarios:
        print("Available scenarios:")
        for name, data in config["scenarios"].items():
            print(f"  {name}: {data['description']}")
        return

    asyncio.run(async_main(args, config, engines))


async def async_main(args, config, engines):
    from .benchmark import CacheBenchmarker
    from .metrics import CacheMetrics
    from .scenarios import (
        generate_shared_system_prompt,
        generate_multi_turn,
        generate_rag_context,
        generate_cache_pressure,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    engine_cfg = engines["engines"][args.engine]
    port = engine_cfg["port"]
    bench_cfg = config["benchmark"]
    benchmarker = CacheBenchmarker(port=port, max_tokens=bench_cfg["max_tokens"])
    metrics = CacheMetrics()
    all_summaries = {}

    scenarios = config["scenarios"]
    if args.scenario:
        scenarios = {k: v for k, v in scenarios.items() if k == args.scenario}

    for scenario_name, scenario_cfg in scenarios.items():
        print(f"\n=== Scenario: {scenario_name} ===")
        print(f"  {scenario_cfg['description']}")

        for caching in [False, True]:
            label = "ON" if caching else "OFF"
            print(f"  Running with prefix caching {label}...")

            if scenario_name == "shared_system_prompt":
                requests = generate_shared_system_prompt(
                    num_requests=scenario_cfg["num_requests"],
                    system_tokens=scenario_cfg["system_prompt_tokens"],
                    user_tokens=scenario_cfg["user_message_tokens"],
                )
                results = await benchmarker.run_scenario_shared_system(
                    requests, scenario_name, caching, bench_cfg["concurrency"]
                )
            elif scenario_name == "multi_turn":
                convos = generate_multi_turn(
                    num_conversations=scenario_cfg["num_conversations"],
                    turns=scenario_cfg["turns_per_conversation"],
                    tokens_per_turn=scenario_cfg["tokens_per_turn"],
                )
                results = await benchmarker.run_scenario_multi_turn(
                    convos, scenario_name, caching
                )
            elif scenario_name == "rag_common_context":
                requests = generate_rag_context(
                    context_tokens=scenario_cfg["context_tokens"],
                    num_queries=scenario_cfg["num_queries"],
                    query_tokens=scenario_cfg["query_tokens"],
                )
                results = await benchmarker.run_scenario_shared_system(
                    requests, scenario_name, caching, bench_cfg["concurrency"]
                )
            elif scenario_name == "cache_pressure":
                batches = generate_cache_pressure(
                    prefix_tokens=scenario_cfg["prefix_tokens"],
                    unique_prefixes=scenario_cfg["unique_prefixes"],
                    requests_per_prefix=scenario_cfg["requests_per_prefix"],
                )
                pressure_results = await benchmarker.run_scenario_cache_pressure(
                    batches, scenario_name, caching, bench_cfg["concurrency"]
                )
                results = []
                for prefix_results in pressure_results.values():
                    results.extend(prefix_results)
            else:
                print(f"  Unknown scenario: {scenario_name}, skipping")
                continue

            for r in results:
                metrics.record(r)

            summary = metrics.summary(scenario_name, caching)
            cache_key = "caching_on" if caching else "caching_off"
            all_summaries.setdefault(scenario_name, {})[cache_key] = summary
            print(f"    TTFT p50: {summary.get('ttft_p50', 0):.1f}ms ({summary['count']} requests)")

        speedup = metrics.ttft_speedup(scenario_name)
        print(f"  Speedup: {speedup:.1f}x")

    results_file = output / "prefix_caching_results.json"
    with open(results_file, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    from .visualize import plot_ttft_comparison, plot_cache_pressure
    plot_ttft_comparison(all_summaries, str(output))
    print(f"Charts saved to {output}/")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

```bash
cd 03-prefix-caching && python -m pytest tests/test_main.py -v
```
Expected: 2 passed

**Step 5: Commit**

```bash
git add 03-prefix-caching/src/main.py 03-prefix-caching/tests/test_main.py
git commit -m "feat(prefix-caching): add CLI entrypoint with all 4 scenarios"
```

---

### Task 17: Write GPU scripts for prefix caching

**Files:**
- Create: `03-prefix-caching/scripts/setup_gpu.sh`
- Create: `03-prefix-caching/scripts/run_benchmarks.sh`

**Step 1: Write scripts**

`03-prefix-caching/scripts/setup_gpu.sh`:
```bash
#!/bin/bash
set -e

echo "=== Setting up prefix caching project on GPU instance ==="

if [ -d "/workspace" ]; then
    mkdir -p /workspace/hf_cache
    ln -sf /workspace/hf_cache /root/.cache/huggingface
fi

pip install --upgrade pip
pip install vllm>=0.16.0
pip install httpx pandas matplotlib pyyaml tqdm
pip install pytest pytest-asyncio

echo "Downloading model..."
python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
"

echo "=== Setup complete ==="
```

`03-prefix-caching/scripts/run_benchmarks.sh`:
```bash
#!/bin/bash
set -e

cd "$(dirname "$0")/.."

ENGINE="${1:-vllm}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8010

echo "=== Running prefix caching benchmarks with $ENGINE ==="

# Run WITHOUT prefix caching
echo "Starting $ENGINE WITHOUT prefix caching..."
if [ "$ENGINE" = "vllm" ]; then
    vllm serve "$MODEL" --port "$PORT" --max-model-len 8192 &
elif [ "$ENGINE" = "sglang" ]; then
    python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --context-length 8192 --disable-radix-cache &
fi
SERVER_PID=$!

echo "Waiting for server..."
sleep 30
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    sleep 5
done

python -m src.main --engine "$ENGINE" 2>&1 | tee results/run_no_cache.log

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 5

# Run WITH prefix caching
echo "Starting $ENGINE WITH prefix caching..."
if [ "$ENGINE" = "vllm" ]; then
    vllm serve "$MODEL" --port "$PORT" --max-model-len 8192 --enable-prefix-caching &
elif [ "$ENGINE" = "sglang" ]; then
    python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --context-length 8192 &
fi
SERVER_PID=$!

echo "Waiting for server..."
sleep 30
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    sleep 5
done

python -m src.main --engine "$ENGINE" 2>&1 | tee results/run_with_cache.log

kill $SERVER_PID 2>/dev/null || true

echo "=== Benchmarks complete ==="
```

**Step 2: Make executable and commit**

```bash
chmod +x 03-prefix-caching/scripts/setup_gpu.sh 03-prefix-caching/scripts/run_benchmarks.sh
git add 03-prefix-caching/scripts/
git commit -m "feat(prefix-caching): add GPU setup and benchmark scripts"
```

---

## Phase 3: Cross-Cutting Concerns

### Task 18: Add Dockerfiles to all projects

**Files:**
- Create: `01-quantization/Dockerfile`
- Create: `02-inference-benchmarks/Dockerfile`
- Create: `03-prefix-caching/Dockerfile`

**Step 1: Write Dockerfiles**

`01-quantization/Dockerfile`:
```dockerfile
FROM vllm/vllm-openai:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install auto-gptq autoawq llmcompressor

COPY . .

ENTRYPOINT ["python", "-m", "src.main"]
```

`02-inference-benchmarks/Dockerfile`:
```dockerfile
FROM vllm/vllm-openai:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "-m", "src.main"]
```

`03-prefix-caching/Dockerfile`:
```dockerfile
FROM vllm/vllm-openai:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "-m", "src.main"]
```

**Step 2: Commit**

```bash
git add 01-quantization/Dockerfile 02-inference-benchmarks/Dockerfile 03-prefix-caching/Dockerfile
git commit -m "feat: add Dockerfiles to all three projects"
```

---

### Task 19: Update root README

**Files:**
- Modify: `README.md`

**Step 1: Write new README**

Replace the entire contents of `README.md` with:

```markdown
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
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update root README for redesigned portfolio"
```

---

### Task 20: Run all tests across the portfolio

**Step 1: Run project 01 tests**

```bash
cd 01-quantization && python -m pytest tests/ -v
```
Expected: All passed

**Step 2: Run project 02 tests**

```bash
cd 02-inference-benchmarks && python -m pytest tests/ -v
```
Expected: All passed (existing tests unchanged)

**Step 3: Run project 03 tests**

```bash
cd 03-prefix-caching && python -m pytest tests/ -v
```
Expected: All passed

**Step 4: Commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: test corrections from full portfolio test run"
```

---

## Phase 4: GPU Execution (RunPod)

> **Note:** This phase requires spinning up a RunPod GPU instance. Budget ~$3-5 for an A40 session (~2-3 hours).

### Task 21: Run quantization pipeline on GPU

**Step 1: Spin up RunPod A40 (48GB)**

**Step 2: Clone repo and setup**

```bash
git clone <repo-url>
cd inference-engineering-portfolio/01-quantization
bash scripts/setup_gpu.sh
```

**Step 3: Run quantization**

```bash
python -m src.main --step quantize
```

**Step 4: Run quality evaluation**

```bash
python -m src.main --step evaluate
```

**Step 5: For each format, start vLLM and benchmark**

```bash
# BF16 baseline
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8010 --max-model-len 4096 &
sleep 60
python -m src.main --step benchmark --format bf16
kill %1; sleep 5

# GPTQ INT4
vllm serve quantized_models/Qwen2.5-7B-Instruct-gptq_int4 --port 8010 --max-model-len 4096 &
sleep 60
python -m src.main --step benchmark --format gptq_int4
kill %1; sleep 5

# AWQ INT4
vllm serve quantized_models/Qwen2.5-7B-Instruct-awq_int4 --port 8010 --max-model-len 4096 &
sleep 60
python -m src.main --step benchmark --format awq_int4
kill %1; sleep 5

# FP8
vllm serve quantized_models/Qwen2.5-7B-Instruct-fp8 --port 8010 --max-model-len 4096 &
sleep 60
python -m src.main --step benchmark --format fp8
kill %1; sleep 5
```

**Step 6: Generate charts**

```bash
python -m src.main --step visualize
```

**Step 7: Commit results**

```bash
git add results/
git commit -m "data(quantization): add GPU benchmark results"
```

### Task 22: Run prefix caching benchmarks on GPU

**Step 1: On same GPU instance, setup project 03**

```bash
cd ../03-prefix-caching
bash scripts/setup_gpu.sh
```

**Step 2: Run benchmarks**

```bash
bash scripts/run_benchmarks.sh vllm
```

**Step 3: Commit results**

```bash
git add results/
git commit -m "data(prefix-caching): add prefix caching benchmark results"
```

### Task 23: Write project READMEs with real results

After GPU data is collected, write detailed READMEs for both new projects. Follow the pattern from `02-inference-benchmarks/README.md`:
- Architecture diagram (ASCII)
- Key results table with real numbers from the GPU runs
- Design Decisions section
- Reproducible quick-start instructions

**Files:**
- Create: `01-quantization/README.md`
- Create: `03-prefix-caching/README.md`

**Step 1: Write READMEs using actual results from the GPU runs**

Template for each: fill in real numbers from `results/*.json`.

**Step 2: Final commit**

```bash
git add 01-quantization/README.md 03-prefix-caching/README.md
git commit -m "docs: add READMEs with real GPU benchmark results"
```

### Task 24: Push and verify

```bash
git push origin main
```

Verify on GitHub that all three projects display correctly with their READMEs.
