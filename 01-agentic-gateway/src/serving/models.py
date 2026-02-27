"""
Model configuration and registry.

Loads model configs from YAML and provides a clean interface
for the rest of the application to reference models.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    name: str
    max_model_len: int
    gpu_memory_utilization: float
    quantization: str | None
    description: str


@dataclass
class CostConfig:
    input_per_1m: float
    output_per_1m: float

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_1m + output_tokens * self.output_per_1m) / 1_000_000


class ModelRegistry:
    """Loads and holds model + cost configs from models.yaml."""

    def __init__(self, config_path: str = "configs/models.yaml"):
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        self.models: dict[str, ModelConfig] = {}
        for key, cfg in raw["models"].items():
            self.models[key] = ModelConfig(
                name=cfg["name"],
                max_model_len=cfg["max_model_len"],
                gpu_memory_utilization=cfg["gpu_memory_utilization"],
                quantization=cfg.get("quantization"),
                description=cfg.get("description", ""),
            )

        self.costs: dict[str, CostConfig] = {}
        for key, cost in raw.get("costs", {}).items():
            self.costs[key] = CostConfig(
                input_per_1m=cost["input"],
                output_per_1m=cost["output"],
            )

    def get(self, model_key: str) -> ModelConfig:
        return self.models[model_key]

    def get_cost(self, model_key: str) -> CostConfig:
        return self.costs[model_key]

    @property
    def available_models(self) -> list[str]:
        return list(self.models.keys())
