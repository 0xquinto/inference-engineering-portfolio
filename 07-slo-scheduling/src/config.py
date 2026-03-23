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
