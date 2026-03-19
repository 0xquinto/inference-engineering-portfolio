import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SpecMethod:
    name: str
    description: str
    spec_type: str | None = None
    draft_model: str | None = None
    num_speculative_tokens: int | None = None
    ngram_prompt_lookup_max: int | None = None
    ngram_prompt_lookup_min: int | None = None
    parallel_drafting: bool = False

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "SpecMethod":
        return cls(
            name=name,
            description=data.get("description", ""),
            spec_type=data.get("spec_type"),
            draft_model=data.get("draft_model"),
            num_speculative_tokens=data.get("num_speculative_tokens"),
            ngram_prompt_lookup_max=data.get("ngram_prompt_lookup_max"),
            ngram_prompt_lookup_min=data.get("ngram_prompt_lookup_min"),
            parallel_drafting=data.get("parallel_drafting", False),
        )

    @property
    def is_baseline(self) -> bool:
        return self.spec_type is None

    def vllm_args(self) -> list[str]:
        if self.is_baseline:
            return []

        args = []

        if self.spec_type == "eagle":
            if self.draft_model:
                args.extend(["--speculative-model", self.draft_model])
            else:
                args.extend(["--speculative-model", "[eagle_head]"])
            if self.num_speculative_tokens is not None:
                args.extend(["--num-speculative-tokens", str(self.num_speculative_tokens)])
            if self.parallel_drafting:
                spec_cfg = json.dumps({"parallel_drafting": True})
                args.extend(["--speculative-config", spec_cfg])

        elif self.spec_type == "ngram":
            args.extend(["--speculative-model", "[ngram]"])
            if self.num_speculative_tokens is not None:
                args.extend(["--num-speculative-tokens", str(self.num_speculative_tokens)])
            if self.ngram_prompt_lookup_max is not None:
                args.extend(["--ngram-prompt-lookup-max", str(self.ngram_prompt_lookup_max)])
            if self.ngram_prompt_lookup_min is not None:
                args.extend(["--ngram-prompt-lookup-min", str(self.ngram_prompt_lookup_min)])

        elif self.spec_type == "draft_model":
            if self.draft_model:
                args.extend(["--speculative-model", self.draft_model])
            if self.num_speculative_tokens is not None:
                args.extend(["--num-speculative-tokens", str(self.num_speculative_tokens)])

        elif self.spec_type == "mtp":
            args.extend(["--speculative-model", "[mtp]"])
            if self.num_speculative_tokens is not None:
                args.extend(["--num-speculative-tokens", str(self.num_speculative_tokens)])

        return args


@dataclass
class SpecConfig:
    model_name: str
    methods: list[SpecMethod]
    benchmark_prompts: list[str]
    qps_levels: list[int]
    port: int = 8010
    warmup_requests: int = 5
    requests_per_prompt: int = 10
    max_tokens: int = 256
    temperature: float = 0.0


def load_config(path: Path) -> SpecConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    methods = [
        SpecMethod.from_dict(name, data)
        for name, data in raw["methods"].items()
    ]

    bench = raw["benchmark"]

    return SpecConfig(
        model_name=raw["model"]["name"],
        methods=methods,
        benchmark_prompts=bench.get("prompts", []),
        qps_levels=bench.get("qps_levels", [1, 5, 10, 25, 50]),
        port=bench.get("port", 8010),
        warmup_requests=bench.get("warmup_requests", 5),
        requests_per_prompt=bench.get("requests_per_prompt", 10),
        max_tokens=bench.get("max_tokens", 256),
        temperature=bench.get("temperature", 0.0),
    )
