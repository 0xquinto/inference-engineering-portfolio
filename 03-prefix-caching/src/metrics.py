from dataclasses import dataclass


@dataclass
class RequestMetric:
    scenario: str
    caching: bool
    request_idx: int
    ttft_ms: float
    total_ms: float
    tokens: int

    @property
    def tokens_per_sec(self) -> float:
        if self.total_ms <= 0:
            return 0.0
        return self.tokens / (self.total_ms / 1000)

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "caching": self.caching,
            "request_idx": self.request_idx,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "tokens": self.tokens,
            "tokens_per_sec": self.tokens_per_sec,
        }


class CacheMetrics:
    def __init__(self):
        self._records: list[RequestMetric] = []

    def record(self, metric: RequestMetric) -> None:
        self._records.append(metric)

    def _filter(self, scenario: str, caching: bool) -> list[RequestMetric]:
        return [r for r in self._records if r.scenario == scenario and r.caching == caching]

    def summary(self, scenario: str, caching: bool) -> dict:
        records = self._filter(scenario, caching)
        if not records:
            return {"count": 0}

        ttfts = sorted(r.ttft_ms for r in records)
        tps_list = sorted(r.tokens_per_sec for r in records)
        n = len(records)

        return {
            "count": n,
            "caching": caching,
            "ttft_p50": ttfts[n // 2],
            "ttft_p95": ttfts[int(n * 0.95)] if n >= 20 else ttfts[-1],
            "ttft_mean": sum(ttfts) / n,
            "throughput_p50": tps_list[n // 2],
        }

    def ttft_speedup(self, scenario: str) -> float:
        off = self.summary(scenario, caching=False)
        on = self.summary(scenario, caching=True)
        if on["count"] == 0 or off["count"] == 0 or on["ttft_p50"] == 0:
            return 0.0
        return off["ttft_p50"] / on["ttft_p50"]
