from dataclasses import dataclass
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
    model_id: str = "default"
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
        model_id=bench.get("model_id", "default"),
        warmup_requests=bench.get("warmup_requests", 3),
        requests_per_prompt=bench.get("requests_per_prompt", 5),
        max_tokens=bench.get("max_tokens", 256),
        temperature=bench.get("temperature", 0.0),
        perplexity_max_samples=eval_cfg.get("perplexity", {}).get("max_samples", 500),
        mmlu_num_tasks=eval_cfg.get("mmlu", {}).get("num_tasks", 14),
        mmlu_few_shot=eval_cfg.get("mmlu", {}).get("num_few_shot", 5),
    )
