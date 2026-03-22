from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Backend:
    name: str
    description: str
    guided_decoding_backend: str | None = None

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "Backend":
        return cls(
            name=name,
            description=data.get("description", ""),
            guided_decoding_backend=data.get("guided_decoding_backend"),
        )

    @property
    def is_constrained(self) -> bool:
        return self.guided_decoding_backend is not None


@dataclass
class SchemaLevel:
    name: str
    description: str
    complexity: str

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "SchemaLevel":
        return cls(
            name=name,
            description=data.get("description", ""),
            complexity=data.get("complexity", "low"),
        )


@dataclass
class StructuredConfig:
    model_name: str
    backends: list[Backend]
    schemas: list[SchemaLevel]
    concurrency_levels: list[int]
    port: int = 8010
    model_id: str = "default"
    warmup_requests: int = 3
    requests_per_prompt: int = 10
    max_tokens: int = 512
    temperature: float = 0.0
    max_retries: int = 3
    schema_format: str = "guided_json"  # "guided_json" (vLLM) or "response_format" (OpenAI/Ollama)
    disable_thinking: bool = False


def load_config(path: Path) -> StructuredConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    backends = [
        Backend.from_dict(name, data)
        for name, data in raw["backends"].items()
    ]

    schemas = [
        SchemaLevel.from_dict(name, data)
        for name, data in raw["schemas"].items()
    ]

    bench = raw["benchmark"]

    return StructuredConfig(
        model_name=raw["model"]["name"],
        backends=backends,
        schemas=schemas,
        concurrency_levels=bench.get("concurrency_levels", [1, 10, 50]),
        port=bench.get("port", 8010),
        model_id=bench.get("model_id", "default"),
        warmup_requests=bench.get("warmup_requests", 3),
        requests_per_prompt=bench.get("requests_per_prompt", 10),
        max_tokens=bench.get("max_tokens", 512),
        temperature=bench.get("temperature", 0.0),
        max_retries=bench.get("max_retries", 3),
        schema_format=bench.get("schema_format", "guided_json"),
        disable_thinking=bench.get("disable_thinking", False),
    )
