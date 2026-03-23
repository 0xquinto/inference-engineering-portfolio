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
    max_tokens: int = 512


class WorkloadGenerator:
    def __init__(self, classes: list[WorkloadClass], seed: int | None = None):
        self.classes = classes
        self.rng = random.Random(seed)

    def generate(self, count: int) -> list[WorkloadRequest]:
        weights = [wc.share for wc in self.classes]
        chosen = self.rng.choices(self.classes, weights=weights, k=count)
        now = time.monotonic()
        return [
            WorkloadRequest(
                prompt=wc.prompt,
                request_class=wc.name,
                slo_seconds=wc.slo_seconds,
                priority=PRIORITY_MAP.get(wc.name, 2),
                deadline=now + wc.slo_seconds,
                max_tokens=wc.max_tokens,
            )
            for wc in chosen
        ]
