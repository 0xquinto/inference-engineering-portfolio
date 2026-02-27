"""Tests for model configuration and registry."""

import textwrap
from pathlib import Path

from src.serving.models import CostConfig, ModelRegistry


def test_cost_config_estimate():
    cost = CostConfig(input_per_1m=0.10, output_per_1m=0.10)
    # 1M input + 1M output = 0.10 + 0.10 = 0.20
    assert cost.estimate_cost(1_000_000, 1_000_000) == 0.20


def test_cost_config_zero_tokens():
    cost = CostConfig(input_per_1m=0.80, output_per_1m=0.80)
    assert cost.estimate_cost(0, 0) == 0.0


def test_cost_config_asymmetric():
    cost = CostConfig(input_per_1m=0.10, output_per_1m=0.80)
    result = cost.estimate_cost(1000, 1000)
    expected = (1000 * 0.10 + 1000 * 0.80) / 1_000_000
    assert abs(result - expected) < 1e-10


def test_registry_loads_yaml(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(textwrap.dedent("""\
        models:
          small:
            name: test/small-model
            max_model_len: 2048
            gpu_memory_utilization: 0.4
            quantization: null
            description: "Test small"
          large:
            name: test/large-model
            max_model_len: 4096
            gpu_memory_utilization: 0.5
            quantization: awq
            description: "Test large"
        costs:
          small:
            input: 0.10
            output: 0.10
          large:
            input: 0.80
            output: 0.80
    """))
    registry = ModelRegistry(str(config))
    assert registry.available_models == ["small", "large"]
    assert registry.get("small").name == "test/small-model"
    assert registry.get("large").quantization == "awq"


def test_registry_get_cost(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(textwrap.dedent("""\
        models:
          small:
            name: test/small
            max_model_len: 2048
            gpu_memory_utilization: 0.4
        costs:
          small:
            input: 0.10
            output: 0.10
    """))
    registry = ModelRegistry(str(config))
    cost = registry.get_cost("small")
    assert cost.input_per_1m == 0.10
    assert cost.output_per_1m == 0.10
