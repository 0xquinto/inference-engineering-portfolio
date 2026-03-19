from dataclasses import dataclass, field


@dataclass
class MethodResult:
    method_name: str
    qps: int
    ttft_ms: float
    total_time_ms: float
    tokens_generated: int
    acceptance_rate: float | None = None

    @property
    def tokens_per_sec(self) -> float:
        if self.total_time_ms <= 0:
            return 0.0
        return self.tokens_generated / (self.total_time_ms / 1000)

    def to_dict(self) -> dict:
        return {
            "method_name": self.method_name,
            "qps": self.qps,
            "ttft_ms": self.ttft_ms,
            "total_time_ms": self.total_time_ms,
            "tokens_generated": self.tokens_generated,
            "tokens_per_sec": self.tokens_per_sec,
            "acceptance_rate": self.acceptance_rate,
        }


class SpecMethodTracker:
    def __init__(self):
        self._records: list[MethodResult] = []

    def record(self, result: MethodResult) -> None:
        self._records.append(result)

    def _filter(self, method: str, qps: int) -> list[MethodResult]:
        return [r for r in self._records if r.method_name == method and r.qps == qps]

    def summary(self, method: str, qps: int) -> dict:
        records = self._filter(method, qps)
        if not records:
            return {"count": 0}

        n = len(records)
        ttfts = sorted(r.ttft_ms for r in records)
        tps_list = sorted(r.tokens_per_sec for r in records)

        acceptance_rates = [r.acceptance_rate for r in records if r.acceptance_rate is not None]
        acceptance_rate_mean = (
            sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else None
        )

        return {
            "count": n,
            "ttft_p50": ttfts[n // 2],
            "ttft_p95": ttfts[int(n * 0.95)] if n >= 20 else ttfts[-1],
            "throughput_p50": tps_list[n // 2],
            "acceptance_rate_mean": acceptance_rate_mean,
        }

    def speedup(self, method: str, qps: int) -> float:
        baseline = self.summary("baseline", qps)
        target = self.summary(method, qps)

        if not baseline.get("count") or not target.get("count"):
            return 0.0

        baseline_tps = baseline["throughput_p50"]
        target_tps = target["throughput_p50"]

        if baseline_tps <= 0:
            return 0.0

        return target_tps / baseline_tps
