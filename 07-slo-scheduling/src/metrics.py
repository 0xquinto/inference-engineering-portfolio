from dataclasses import dataclass


@dataclass
class RequestResult:
    request_class: str
    slo_seconds: float
    latency_seconds: float
    ttft_seconds: float
    tokens: int

    @property
    def meets_slo(self) -> bool:
        return self.latency_seconds <= self.slo_seconds


def compute_goodput(results: list[RequestResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.meets_slo) / len(results)


def compute_goodput_per_class(results: list[RequestResult]) -> dict[str, float]:
    if not results:
        return {}
    by_class: dict[str, list[RequestResult]] = {}
    for r in results:
        by_class.setdefault(r.request_class, []).append(r)
    return {
        cls: sum(1 for r in rs if r.meets_slo) / len(rs)
        for cls, rs in by_class.items()
    }


def compute_fairness(per_class_goodput: dict[str, float]) -> float:
    if not per_class_goodput:
        return 0.0
    values = list(per_class_goodput.values())
    return min(values) / max(values) if max(values) > 0 else 0.0


def compute_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(latencies)
    n = len(s)
    return {
        "p50": s[n // 2],
        "p95": s[int(n * 0.95)] if n >= 20 else s[-1],
        "p99": s[int(n * 0.99)] if n >= 100 else s[-1],
    }
