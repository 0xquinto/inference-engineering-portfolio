# Hardware-Agnostic Profile System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every project runnable on both CUDA GPUs and Apple Silicon Macs via a `--profile gpu|local` flag, with no code changes between platforms.

**Architecture:** Profile YAMLs use the **exact same schema** as existing config YAMLs. The `--profile gpu` flag simply means `load_config(Path("profiles/gpu.yaml"))`. No new abstraction layer — profiles are just alternative configs for different hardware. The benchmark code is already backend-agnostic (httpx → OpenAI API). Project 01 additionally gets an MLX quantization runner. Project 06 gets a `$0/hr` local hardware tier for edge-cloud cascade routing.

**Tech Stack:** Python 3.10+, PyYAML, httpx, matplotlib, vLLM (CUDA + Metal auto-detected on macOS), MLX-LM (Mac quantization)

**Key design decision:** Profile YAMLs are full config files, not overlays. This means `profiles/gpu.yaml` is a copy of the existing config, and `profiles/local.yaml` is a smaller variant. The existing `--config` flag still works for backwards compatibility.

**Note on `profiles.py`:** The profile loader is a 7-line function identical across all 6 projects. This is intentional — each project is self-contained with no shared package. Copy verbatim from Task 1, Step 5.

**Note on vLLM Metal:** On macOS, `pip install vllm` auto-detects Apple Silicon and uses the Metal backend. There is no separate `vllm-metal` package. The engine name in configs remains `vllm` — same binary, different backend.

---

## File Structure

Each of the 6 projects gains these files:

```
<project>/
├── profiles/
│   ├── gpu.yaml          # full config for CUDA (same schema as configs/*.yaml)
│   └── local.yaml        # full config for Apple Silicon (smaller models, lower concurrency)
├── scripts/
│   ├── setup_gpu.sh      # existing
│   └── setup_local.sh    # NEW: installs vllm + mlx-lm on Mac
└── src/
    └── profiles.py       # NEW: thin profile loader (7 lines, identical across projects)
```

Project 01 additionally gets:
```
01-quantization/src/quantize_mlx.py   # NEW: MLX quantization runner
```

---

### Task 1: Project 04 — Speculative Decoding Profile System (reference implementation)

Build the profile system in project 04 first. Then replicate the pattern across all projects.

**Files:**
- Create: `04-speculative-decoding/profiles/gpu.yaml`
- Create: `04-speculative-decoding/profiles/local.yaml`
- Create: `04-speculative-decoding/src/profiles.py`
- Modify: `04-speculative-decoding/src/main.py`
- Create: `04-speculative-decoding/scripts/setup_local.sh`
- Test: `04-speculative-decoding/tests/test_profiles.py`

- [ ] **Step 1: Write the failing test for profile loader**

```python
# tests/test_profiles.py
from pathlib import Path
from src.profiles import load_profile


class TestLoadProfile:
    def test_load_gpu_profile(self):
        profile = load_profile("gpu", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-9B"
        assert profile["benchmark"]["qps_levels"] == [1, 5, 10, 25, 50]

    def test_load_local_profile(self):
        profile = load_profile("local", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-4B"
        assert max(profile["benchmark"]["qps_levels"]) <= 10

    def test_invalid_profile_raises(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent", Path(__file__).parent.parent / "profiles")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 04-speculative-decoding && python -m pytest tests/test_profiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.profiles'`

- [ ] **Step 3: Create profiles/gpu.yaml**

Copy `configs/speculative.yaml` verbatim as `profiles/gpu.yaml`.

- [ ] **Step 4: Create profiles/local.yaml**

Same schema as `configs/speculative.yaml` but with smaller model and reduced params:

```yaml
# profiles/local.yaml — Apple Silicon (M4 MacBook Pro, 24GB unified memory)
model:
  name: Qwen/Qwen3.5-4B
  type: dense
  params: 4B

methods:
  baseline:
    description: "Autoregressive decoding (no speculation)"
    spec_type: null
  ngram:
    description: "N-gram suffix matching (no extra model)"
    spec_type: ngram
    ngram_prompt_lookup_max: 5
    ngram_prompt_lookup_min: 2
    num_speculative_tokens: 5
  draft_model:
    description: "Small draft model verification"
    spec_type: draft_model
    draft_model: Qwen/Qwen3.5-0.8B
    num_speculative_tokens: 5

benchmark:
  port: 8010
  qps_levels: [1, 5, 10]
  warmup_requests: 2
  requests_per_prompt: 3
  max_tokens: 128
  temperature: 0.0
  prompts:
    - "What is the capital of France?"
    - "Explain quantum computing in simple terms."
    - "Write a Python function to check if a number is prime."
```

- [ ] **Step 5: Create src/profiles.py**

```python
from pathlib import Path

import yaml


def load_profile(name: str, profiles_dir: Path | None = None) -> dict:
    """Load a hardware profile by name (gpu or local).

    Profiles are YAML files in the profiles/ directory with the same
    schema as configs/*.yaml. This allows reuse of the existing
    load_config() function.
    """
    if profiles_dir is None:
        profiles_dir = Path("profiles")
    path = profiles_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
```

- [ ] **Step 6: Update src/main.py to accept --profile**

The key insight: profile YAMLs have the same schema as config YAMLs. So `--profile` just changes which file path `load_config` receives:

```python
# In argparse setup, add:
parser.add_argument("--profile", choices=["gpu", "local"],
                    help="Hardware profile (gpu or local)")

# Replace the config loading line:
if args.profile:
    config_path = Path(f"profiles/{args.profile}.yaml")
else:
    config_path = Path(args.config)
cfg = load_config(config_path)
```

This is a 4-line change to `main()`. No config construction duplication.

- [ ] **Step 7: Run all tests to verify nothing breaks**

Run: `cd 04-speculative-decoding && python -m pytest tests/ -v`
Expected: All existing tests pass + 3 new profile tests pass.

- [ ] **Step 8: Create scripts/setup_local.sh**

```bash
#!/bin/bash
set -e

echo "=== Setting up speculative decoding project on Apple Silicon ==="

pip install --upgrade pip
pip install vllm  # Metal backend auto-detected on macOS
pip install httpx pandas matplotlib pyyaml tqdm
pip install pytest pytest-asyncio

python -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen3.5-4B...')
snapshot_download('Qwen/Qwen3.5-4B')
print('Downloading Qwen3.5-0.8B (draft model)...')
snapshot_download('Qwen/Qwen3.5-0.8B')
print('Done.')
"

echo "=== Setup complete. Start server with: ==="
echo "vllm serve Qwen/Qwen3.5-4B --port 8010"
echo "Then run: python -m src.main --profile local"
```

- [ ] **Step 9: Commit**

```bash
git add 04-speculative-decoding/profiles/ 04-speculative-decoding/src/profiles.py \
       04-speculative-decoding/tests/test_profiles.py 04-speculative-decoding/src/main.py \
       04-speculative-decoding/scripts/setup_local.sh
git commit -m "feat(speculative-decoding): add hardware profile system (gpu/local)"
```

---

### Task 2: Project 01 — Quantization Pipeline (profiles + MLX runner)

Project 01 needs profiles AND a second quantization backend (MLX) for Apple Silicon.

**Files:**
- Create: `01-quantization/profiles/gpu.yaml`
- Create: `01-quantization/profiles/local.yaml`
- Create: `01-quantization/src/profiles.py` (copy verbatim from Task 1 Step 5)
- Create: `01-quantization/src/quantize_mlx.py`
- Modify: `01-quantization/src/quantize.py` (add `mlx` to dispatch)
- Modify: `01-quantization/src/main.py`
- Create: `01-quantization/scripts/setup_local.sh`
- Test: `01-quantization/tests/test_profiles.py`
- Test: `01-quantization/tests/test_quantize_mlx.py`

- [ ] **Step 1: Write failing test for profile loader**

```python
# tests/test_profiles.py
from pathlib import Path
from src.profiles import load_profile


class TestLoadProfile:
    def test_load_gpu_profile(self):
        profile = load_profile("gpu", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-9B"

    def test_load_local_profile(self):
        profile = load_profile("local", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-4B"
        assert profile["formats"]["int4"]["tool"] == "mlx"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 01-quantization && python -m pytest tests/test_profiles.py -v`

- [ ] **Step 3: Create profiles/gpu.yaml**

Copy `configs/quantization.yaml` verbatim as `profiles/gpu.yaml`.

- [ ] **Step 4: Create profiles/local.yaml**

```yaml
# profiles/local.yaml — Apple Silicon quantization via MLX
model:
  name: Qwen/Qwen3.5-4B
  type: dense
  params: 4B
  base_precision: bf16

formats:
  bf16:
    description: "BF16 baseline (no quantization)"
    tool: null
  int4:
    description: "INT4 weight-only quantization via MLX"
    tool: mlx
    bits: 4
    group_size: 64
  int8:
    description: "INT8 quantization via MLX"
    tool: mlx
    bits: 8
    group_size: 64

evaluation:
  perplexity:
    dataset: wikitext
    subset: wikitext-2-raw-v1
    split: test
    max_samples: 100
  mmlu:
    num_tasks: 5
    num_few_shot: 5
    max_samples_per_task: 50

benchmark:
  engine: vllm
  port: 8010
  concurrency_levels: [1, 5, 10]
  warmup_requests: 2
  requests_per_prompt: 3
  max_tokens: 128
  temperature: 0.0
  prompts:
    - "What is the capital of France?"
    - "Explain quantum computing in simple terms."
```

- [ ] **Step 5: Create src/profiles.py** (copy verbatim from Task 1 Step 5)

- [ ] **Step 6: Write failing test for MLX quantization runner**

```python
# tests/test_quantize_mlx.py
import pytest
from unittest.mock import patch
from src.quantize_mlx import MLXQuantizationRunner
from src.config import QuantFormat


class TestMLXQuantizationRunner:
    def test_init(self):
        runner = MLXQuantizationRunner(model_name="Qwen/Qwen3.5-4B")
        assert runner.model_name == "Qwen/Qwen3.5-4B"

    def test_skip_baseline(self):
        runner = MLXQuantizationRunner(model_name="Qwen/Qwen3.5-4B")
        fmt = QuantFormat.from_dict("bf16", {"description": "baseline", "tool": None})
        result = runner.quantize(fmt)
        assert result.format_name == "bf16"
        assert result.time_seconds == 0.0

    @patch("src.quantize_mlx.mlx_lm_convert")
    def test_quantize_int4(self, mock_convert):
        mock_convert.return_value = None
        runner = MLXQuantizationRunner(model_name="Qwen/Qwen3.5-4B")
        fmt = QuantFormat.from_dict("int4", {
            "description": "INT4 via MLX",
            "tool": "mlx",
            "bits": 4,
            "group_size": 64,
        })
        result = runner.quantize(fmt)
        assert result.format_name == "int4"
        mock_convert.assert_called_once()
```

- [ ] **Step 7: Run test to verify it fails**

Run: `cd 01-quantization && python -m pytest tests/test_quantize_mlx.py -v`

- [ ] **Step 8: Create src/quantize_mlx.py**

```python
import time
from pathlib import Path

from .config import QuantFormat
from .quantize import QuantizeResult, _dir_size_mb


def mlx_lm_convert(model: str, output_dir: str, bits: int, group_size: int) -> None:
    """Wrapper around mlx_lm.convert. Imported lazily."""
    from mlx_lm import convert
    convert(model=model, quantize=True, q_bits=bits, q_group_size=group_size,
            upload_repo=None, mlx_path=output_dir)


class MLXQuantizationRunner:
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

        output_path = self._output_path(fmt)
        bits = fmt.bits or 4
        group_size = fmt.group_size or 64

        start = time.time()
        mlx_lm_convert(
            model=self.model_name,
            output_dir=str(output_path),
            bits=bits,
            group_size=group_size,
        )
        elapsed = time.time() - start

        return QuantizeResult(
            format_name=fmt.name,
            output_path=str(output_path),
            time_seconds=elapsed,
            original_size_mb=0,
            quantized_size_mb=_dir_size_mb(output_path),
        )

    def _output_path(self, fmt: QuantFormat) -> Path:
        short_name = self.model_name.split("/")[-1]
        return self.output_dir / f"{short_name}-{fmt.name}-mlx"
```

- [ ] **Step 9: Update src/quantize.py dispatch**

Add `"mlx"` to the dispatch dict in `QuantizationRunner.quantize()`. The lazy import means MLX deps only load when the `mlx` tool is actually used:

```python
dispatch = {
    "llm_compressor": self._run_llmcompressor,
    "mlx": self._run_mlx,
}
```

Add method:
```python
def _run_mlx(self, fmt: QuantFormat) -> QuantizeResult:
    from .quantize_mlx import MLXQuantizationRunner
    mlx_runner = MLXQuantizationRunner(self.model_name, str(self.output_dir))
    return mlx_runner.quantize(fmt)
```

- [ ] **Step 10: Update src/main.py to accept --profile**

Same pattern as Task 1 Step 6: resolve profile path, pass to existing `load_config`.

```python
parser.add_argument("--profile", choices=["gpu", "local"],
                    help="Hardware profile (gpu or local)")

# Config loading:
if args.profile:
    config_path = Path(f"profiles/{args.profile}.yaml")
else:
    config_path = Path(args.config)
cfg = load_config(config_path)
```

- [ ] **Step 11: Create scripts/setup_local.sh**

```bash
#!/bin/bash
set -e

echo "=== Setting up quantization project on Apple Silicon ==="

pip install --upgrade pip
pip install mlx-lm
pip install vllm  # Metal backend auto-detected on macOS
pip install httpx pandas matplotlib pyyaml tqdm
pip install pytest pytest-asyncio

python -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen3.5-4B...')
snapshot_download('Qwen/Qwen3.5-4B')
print('Done.')
"

echo "=== Setup complete ==="
echo "Run: python -m src.main --profile local --step quantize"
```

- [ ] **Step 12: Run all tests**

Run: `cd 01-quantization && python -m pytest tests/ -v`
Expected: All existing tests pass + new profile and MLX tests pass.

- [ ] **Step 13: Commit**

```bash
git add 01-quantization/profiles/ 01-quantization/src/profiles.py \
       01-quantization/src/quantize_mlx.py 01-quantization/src/quantize.py \
       01-quantization/src/main.py 01-quantization/scripts/setup_local.sh \
       01-quantization/tests/test_profiles.py 01-quantization/tests/test_quantize_mlx.py
git commit -m "feat(quantization): add hardware profiles + MLX quantization backend"
```

---

### Task 3: Project 02 — Inference Benchmarks Profile System

Project 02 uses raw dicts (not typed dataclasses) for config. The `--profile` integration is simpler.

**Files:**
- Create: `02-inference-benchmarks/profiles/gpu.yaml`
- Create: `02-inference-benchmarks/profiles/local.yaml`
- Create: `02-inference-benchmarks/src/profiles.py` (copy from Task 1 Step 5)
- Modify: `02-inference-benchmarks/src/main.py`
- Create: `02-inference-benchmarks/scripts/setup_local.sh`
- Test: `02-inference-benchmarks/tests/test_profiles.py`

- [ ] **Step 1: Write failing test for profile loader**

```python
# tests/test_profiles.py
from pathlib import Path
from src.profiles import load_profile


def test_load_gpu_profile():
    profile = load_profile("gpu", Path(__file__).parent.parent / "profiles")
    assert "engines" in profile

def test_load_local_profile():
    profile = load_profile("local", Path(__file__).parent.parent / "profiles")
    assert profile["model"]["name"] == "Qwen/Qwen3.5-4B"
    assert "vllm" in profile["engines"]
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Create profiles/gpu.yaml**

Copy `configs/engines.yaml` verbatim as `profiles/gpu.yaml`.

- [ ] **Step 4: Create profiles/local.yaml**

```yaml
# profiles/local.yaml — Apple Silicon benchmark
model:
  name: Qwen/Qwen3.5-4B
  type: dense

engines:
  vllm:
    port: 8010
    extra_args:
      - "--max-model-len"
      - "4096"

benchmark:
  concurrency_levels: [1, 5, 10]
  requests_per_prompt: 3
  max_tokens: 128
```

- [ ] **Step 5: Create src/profiles.py** (copy from Task 1 Step 5)

- [ ] **Step 6: Update src/main.py with --profile flag**

Project 02 uses `load_config()` which returns a raw dict, and `load_prompts()` which loads prompts from a separate JSON. The integration:

```python
# In argparse, add:
parser.add_argument("--profile", choices=["gpu", "local"],
                    help="Hardware profile (gpu or local)")

# In async_main, replace config loading:
if args.profile:
    from .profiles import load_profile
    config = load_profile(args.profile)
else:
    config = load_config(args.config)
prompts = load_prompts(args.prompts)
```

- [ ] **Step 7: Create scripts/setup_local.sh**

```bash
#!/bin/bash
set -e

echo "=== Setting up inference benchmarks on Apple Silicon ==="

pip install --upgrade pip
pip install vllm
pip install httpx pandas matplotlib plotly pyyaml tqdm nvidia-ml-py
pip install pytest pytest-asyncio

python -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen3.5-4B...')
snapshot_download('Qwen/Qwen3.5-4B')
print('Done.')
"

echo "=== Setup complete ==="
echo "Start vLLM: vllm serve Qwen/Qwen3.5-4B --port 8010"
echo "Then run: python -m src.main --profile local --engine vllm"
```

- [ ] **Step 8: Run all tests, commit**

Run: `python -m pytest 02-inference-benchmarks/ -v`

```bash
git commit -m "feat(benchmarks): add hardware profile system (gpu/local)"
```

---

### Task 4: Project 03 — Prefix Caching Profile System

Project 03 loads TWO config files: `--config` (scenarios) and `--engines`. The profile must combine both into one file.

**Files:**
- Create: `03-prefix-caching/profiles/gpu.yaml`
- Create: `03-prefix-caching/profiles/local.yaml`
- Create: `03-prefix-caching/src/profiles.py` (copy from Task 1 Step 5)
- Modify: `03-prefix-caching/src/main.py`
- Create: `03-prefix-caching/scripts/setup_local.sh`
- Test: `03-prefix-caching/tests/test_profiles.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_profiles.py
from pathlib import Path
from src.profiles import load_profile


class TestLoadProfile:
    def test_load_gpu_profile(self):
        profile = load_profile("gpu", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-9B"
        assert "scenarios" in profile
        assert "engines" in profile

    def test_load_local_profile(self):
        profile = load_profile("local", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-4B"
        assert profile["benchmark"]["concurrency"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Create profiles/gpu.yaml**

Merge `configs/scenarios.yaml` and `configs/engines.yaml` into one file:

```yaml
# profiles/gpu.yaml — merges scenarios.yaml + engines.yaml
model:
  name: Qwen/Qwen3.5-9B
  type: dense
  params: 9B

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

engines:
  vllm:
    port: 8010
    prefix_caching_flag: "--enable-prefix-caching"
    extra_args:
      - "--max-model-len"
      - "8192"
  sglang:
    port: 8020
    prefix_caching_flag: null
    extra_args:
      - "--context-length"
      - "8192"

benchmark:
  concurrency: 10
  warmup_requests: 3
  max_tokens: 128
  temperature: 0.0
```

- [ ] **Step 4: Create profiles/local.yaml**

```yaml
# profiles/local.yaml — Apple Silicon prefix caching
model:
  name: Qwen/Qwen3.5-4B
  type: dense
  params: 4B

scenarios:
  shared_system_prompt:
    description: "50 requests with identical system prompt"
    system_prompt_tokens: 500
    num_requests: 50
    user_message_tokens: 30
  multi_turn:
    description: "5-turn conversations"
    num_conversations: 5
    turns_per_conversation: 5
    tokens_per_turn: 50

engines:
  vllm:
    port: 8010
    prefix_caching_flag: "--enable-prefix-caching"
    extra_args:
      - "--max-model-len"
      - "4096"

benchmark:
  concurrency: 5
  warmup_requests: 2
  max_tokens: 64
  temperature: 0.0
```

- [ ] **Step 5: Create src/profiles.py** (copy from Task 1 Step 5)

- [ ] **Step 6: Update src/main.py with --profile flag**

Project 03 loads two separate config files. When `--profile` is used, both come from the single profile YAML:

```python
# In argparse, add:
parser.add_argument("--profile", choices=["gpu", "local"],
                    help="Hardware profile (gpu or local)")

# In main(), replace config loading:
if args.profile:
    from .profiles import load_profile
    profile = load_profile(args.profile)
    config = profile  # scenarios + benchmark in one file
    engines = profile  # engines section also in same file
else:
    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.engines) as f:
        engines = yaml.safe_load(f)

# asyncio.run(async_main(args, config, engines)) — unchanged
```

- [ ] **Step 7: Create scripts/setup_local.sh**

```bash
#!/bin/bash
set -e

echo "=== Setting up prefix caching project on Apple Silicon ==="

pip install --upgrade pip
pip install vllm
pip install httpx pandas matplotlib pyyaml tqdm
pip install pytest pytest-asyncio

python -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen3.5-4B...')
snapshot_download('Qwen/Qwen3.5-4B')
print('Done.')
"

echo "=== Setup complete ==="
echo "Start vLLM: vllm serve Qwen/Qwen3.5-4B --port 8010 --enable-prefix-caching"
echo "Then run: python -m src.main --profile local"
```

- [ ] **Step 8: Run all tests, commit**

```bash
git commit -m "feat(prefix-caching): add hardware profile system (gpu/local)"
```

---

### Task 5: Project 05 — Structured Output Profile System

**Files:**
- Create: `05-structured-output/profiles/gpu.yaml`
- Create: `05-structured-output/profiles/local.yaml`
- Create: `05-structured-output/src/profiles.py` (copy from Task 1 Step 5)
- Modify: `05-structured-output/src/main.py`
- Create: `05-structured-output/scripts/setup_local.sh`
- Test: `05-structured-output/tests/test_profiles.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_profiles.py
from pathlib import Path
from src.profiles import load_profile


class TestLoadProfile:
    def test_load_gpu_profile(self):
        profile = load_profile("gpu", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-9B"
        assert "xgrammar" in profile["backends"]

    def test_load_local_profile(self):
        profile = load_profile("local", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-4B"
        assert max(profile["benchmark"]["concurrency_levels"]) <= 10
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Create profiles/gpu.yaml**

Copy `configs/structured.yaml` verbatim as `profiles/gpu.yaml`.

- [ ] **Step 4: Create profiles/local.yaml**

```yaml
# profiles/local.yaml — Apple Silicon structured output
model:
  name: Qwen/Qwen3.5-4B
  type: dense
  params: 4B

backends:
  xgrammar:
    description: "XGrammar guided decoding"
    guided_decoding_backend: xgrammar
  unconstrained:
    description: "No grammar enforcement (baseline + retry)"
    guided_decoding_backend: null

schemas:
  simple_json:
    description: "Flat JSON object with 3 string fields"
    complexity: low
  function_call:
    description: "Tool-calling schema with dynamic function selection"
    complexity: high

benchmark:
  engine: vllm
  port: 8010
  concurrency_levels: [1, 5]
  warmup_requests: 2
  requests_per_prompt: 3
  max_tokens: 256
  temperature: 0.0
  max_retries: 3
```

- [ ] **Step 5: Create src/profiles.py** (copy from Task 1 Step 5)

- [ ] **Step 6: Update src/main.py with --profile flag**

Same pattern as Task 1 Step 6:

```python
parser.add_argument("--profile", choices=["gpu", "local"],
                    help="Hardware profile (gpu or local)")

if args.profile:
    config_path = Path(f"profiles/{args.profile}.yaml")
else:
    config_path = Path(args.config)
cfg = load_config(config_path)
```

- [ ] **Step 7: Create scripts/setup_local.sh**

```bash
#!/bin/bash
set -e

echo "=== Setting up structured output project on Apple Silicon ==="

pip install --upgrade pip
pip install vllm
pip install httpx pandas matplotlib pyyaml tqdm
pip install pytest pytest-asyncio

python -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen3.5-4B...')
snapshot_download('Qwen/Qwen3.5-4B')
print('Done.')
"

echo "=== Setup complete ==="
echo "Start vLLM: vllm serve Qwen/Qwen3.5-4B --port 8010"
echo "Then run: python -m src.main --profile local"
```

- [ ] **Step 8: Run all tests, commit**

```bash
git commit -m "feat(structured-output): add hardware profile system (gpu/local)"
```

---

### Task 6: Project 06 — Cost Optimization (profiles + edge-cloud cascade)

The local profile introduces `$0/hr` local tiers for edge-cloud cascade routing.

**Files:**
- Create: `06-cost-optimization/profiles/gpu.yaml`
- Create: `06-cost-optimization/profiles/local.yaml`
- Create: `06-cost-optimization/src/profiles.py` (copy from Task 1 Step 5)
- Modify: `06-cost-optimization/src/main.py`
- Create: `06-cost-optimization/scripts/setup_local.sh`
- Test: `06-cost-optimization/tests/test_profiles.py`
- Modify: `06-cost-optimization/tests/test_cost_model.py` (add zero-cost test)

- [ ] **Step 1: Write failing test for profile loader**

```python
# tests/test_profiles.py
from pathlib import Path
from src.profiles import load_profile


class TestLoadProfile:
    def test_load_gpu_profile(self):
        profile = load_profile("gpu", Path(__file__).parent.parent / "profiles")
        assert profile["models"]["small"]["gpu_cost_per_hour"] == 0.75

    def test_load_local_profile(self):
        profile = load_profile("local", Path(__file__).parent.parent / "profiles")
        assert profile["models"]["local_small"]["gpu_cost_per_hour"] == 0.0
        assert profile["models"]["local_medium"]["gpu_cost_per_hour"] == 0.0
        assert profile["models"]["cloud"]["gpu_cost_per_hour"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Create profiles/gpu.yaml**

Copy `configs/cost.yaml` verbatim as `profiles/gpu.yaml`.

- [ ] **Step 4: Create profiles/local.yaml**

```yaml
# profiles/local.yaml — Edge-cloud hybrid cascade
models:
  local_small:
    name: Qwen/Qwen3.5-0.8B
    params: 0.8B
    description: "On-device fast model (M4 MacBook)"
    port: 8010
    gpu_cost_per_hour: 0.0
    vram_mb: 1600
  local_medium:
    name: Qwen/Qwen3.5-4B
    params: 4B
    description: "On-device balanced model (M4 MacBook)"
    port: 8011
    gpu_cost_per_hour: 0.0
    vram_mb: 8000
  cloud:
    name: Qwen/Qwen3.5-9B
    params: 9B
    description: "Cloud GPU for complex queries"
    port: 8012
    gpu_cost_per_hour: 0.75
    vram_mb: 14200

cascade:
  complexity_keywords:
    simple:
      - "what is"
      - "who is"
      - "define"
      - "capital of"
      - "yes or no"
    moderate:
      - "explain"
      - "compare"
      - "summarize"
      - "describe"
      - "how does"
    complex:
      - "analyze"
      - "design"
      - "implement"
      - "write a function"
      - "evaluate the tradeoffs"
  routing:
    simple: local_small
    moderate: local_medium
    complex: cloud
  quality_threshold: 0.8

benchmark:
  concurrency: 5
  warmup_requests: 2
  requests_per_prompt: 3
  max_tokens: 128
  temperature: 0.0
  prompts:
    simple:
      - "What is the capital of France?"
      - "Define photosynthesis."
    moderate:
      - "Explain the difference between TCP and UDP."
      - "How does a transformer model work?"
    complex:
      - "Design a distributed cache invalidation strategy for a multi-region deployment."

cost_analysis:
  gpu_hours_per_month: 720
  target_utilization: 0.5
  api_comparison:
    openai_gpt4o:
      input_per_million: 2.50
      output_per_million: 10.00
    openai_gpt4o_mini:
      input_per_million: 0.15
      output_per_million: 0.60
    anthropic_sonnet:
      input_per_million: 3.00
      output_per_million: 15.00
```

- [ ] **Step 5: Add zero-cost test to test_cost_model.py**

```python
# Add to tests/test_cost_model.py
def test_local_models_zero_cost():
    """Verify that $0/hr local hardware produces $0/M token cost."""
    from src.cost_model import calculate_cost_per_million_tokens
    cost = calculate_cost_per_million_tokens(tps=100.0, gpu_cost_per_hour=0.0)
    assert cost == 0.0
```

This test passes immediately — `calculate_cost_per_million_tokens` already handles zero cost correctly. The test documents the edge-cloud behavior.

- [ ] **Step 6: Create src/profiles.py** (copy from Task 1 Step 5)

- [ ] **Step 7: Update src/main.py with --profile flag**

Same pattern as Task 1 Step 6:

```python
parser.add_argument("--profile", choices=["gpu", "local"],
                    help="Hardware profile (gpu or local)")

if args.profile:
    config_path = Path(f"profiles/{args.profile}.yaml")
else:
    config_path = Path(args.config)
cfg = load_config(config_path)
```

- [ ] **Step 8: Create scripts/setup_local.sh**

```bash
#!/bin/bash
set -e

echo "=== Setting up cost optimization project on Apple Silicon ==="

pip install --upgrade pip
pip install vllm
pip install httpx pandas matplotlib pyyaml tqdm
pip install pytest pytest-asyncio

python -c "
from huggingface_hub import snapshot_download
print('Downloading model tokenizers...')
snapshot_download('Qwen/Qwen3.5-0.8B')
snapshot_download('Qwen/Qwen3.5-4B')
print('Done. (9B cloud model accessed remotely during benchmark)')
"

echo "=== Setup complete ==="
echo "Start local models:"
echo "  vllm serve Qwen/Qwen3.5-0.8B --port 8010"
echo "  vllm serve Qwen/Qwen3.5-4B --port 8011"
echo "Then run: python -m src.main --profile local"
```

- [ ] **Step 9: Run all tests, commit**

```bash
git commit -m "feat(cost-optimization): add edge-cloud hybrid cascade with local hardware tier"
```

---

### Task 7: Update Root README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add Hardware Profiles section to README**

Add after "Skill Progression":

```markdown
## Hardware Profiles

Every project supports `--profile gpu` and `--profile local`:

| Profile | Hardware | Backend | Model | Use case |
|---------|----------|---------|-------|----------|
| `gpu` | A40/H200 (CUDA) | vLLM / SGLang | Qwen3.5-9B | Production benchmarks |
| `local` | M4 MacBook (Metal) | vLLM (Metal) | Qwen3.5-4B | Development + edge inference |

Same pipeline, same metrics, same charts. Hardware is a config field.

```bash
# GPU (RunPod)
python -m src.main --profile gpu

# Local (MacBook)
python -m src.main --profile local
```

The `--config` flag still works for custom configurations.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add hardware profile documentation to README"
```

---

### Task 8: Final Integration Test

- [ ] **Step 1: Run all tests across all 6 projects from repo root**

```bash
for proj in 01-quantization 02-inference-benchmarks 03-prefix-caching \
            04-speculative-decoding 05-structured-output 06-cost-optimization; do
    echo "=== $proj ===" && python -m pytest "$proj/" -q
done
```

Expected: All tests pass across all 6 projects.

- [ ] **Step 2: Verify --profile flag works in each project**

```bash
cd 04-speculative-decoding && python -m src.main --profile local --list-methods && cd ..
cd 01-quantization && python -m src.main --profile local --list-formats && cd ..
cd 05-structured-output && python -m src.main --profile local --list-backends && cd ..
cd 06-cost-optimization && python -m src.main --profile local --list-models && cd ..
```

- [ ] **Step 3: Final commit and push**

```bash
git push
```
