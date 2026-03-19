from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelTier:
    name: str
    model_name: str
    params: str
    description: str
    port: int
    gpu_cost_per_hour: float
    vram_mb: int

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "ModelTier":
        return cls(
            name=name,
            model_name=data["name"],
            params=str(data.get("params", "")),
            description=data.get("description", ""),
            port=data.get("port", 8010),
            gpu_cost_per_hour=data.get("gpu_cost_per_hour", 0.0),
            vram_mb=data.get("vram_mb", 0),
        )


@dataclass
class CascadeConfig:
    complexity_keywords: dict[str, list[str]]
    routing: dict[str, str]
    quality_threshold: float = 0.8


@dataclass
class CostAnalysisConfig:
    gpu_hours_per_month: int = 720
    target_utilization: float = 0.5
    api_comparison: dict = field(default_factory=dict)


@dataclass
class CostConfig:
    models: list[ModelTier]
    cascade: CascadeConfig
    cost_analysis: CostAnalysisConfig
    prompts: dict[str, list[str]] = field(default_factory=dict)
    concurrency: int = 10
    warmup_requests: int = 3
    requests_per_prompt: int = 5
    max_tokens: int = 256
    temperature: float = 0.0


def load_config(path: Path) -> CostConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    models = [
        ModelTier.from_dict(name, data)
        for name, data in raw["models"].items()
    ]

    cascade_raw = raw["cascade"]
    cascade = CascadeConfig(
        complexity_keywords=cascade_raw["complexity_keywords"],
        routing=cascade_raw["routing"],
        quality_threshold=cascade_raw.get("quality_threshold", 0.8),
    )

    cost_raw = raw.get("cost_analysis", {})
    cost_analysis = CostAnalysisConfig(
        gpu_hours_per_month=cost_raw.get("gpu_hours_per_month", 720),
        target_utilization=cost_raw.get("target_utilization", 0.5),
        api_comparison=cost_raw.get("api_comparison", {}),
    )

    bench = raw.get("benchmark", {})

    return CostConfig(
        models=models,
        cascade=cascade,
        cost_analysis=cost_analysis,
        prompts=bench.get("prompts", {}),
        concurrency=bench.get("concurrency", 10),
        warmup_requests=bench.get("warmup_requests", 3),
        requests_per_prompt=bench.get("requests_per_prompt", 5),
        max_tokens=bench.get("max_tokens", 256),
        temperature=bench.get("temperature", 0.0),
    )
