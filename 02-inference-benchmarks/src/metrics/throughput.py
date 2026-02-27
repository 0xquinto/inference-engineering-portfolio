"""Throughput calculation: tokens/sec and requests/sec."""


class ThroughputCalculator:
    """Tracks request and token throughput."""

    def __init__(self):
        self._requests: list[dict] = []

    def record_request(self, total_ms: float, tokens_generated: int):
        self._requests.append({
            "total_ms": total_ms,
            "tokens": tokens_generated,
        })

    def summary(self) -> dict:
        if not self._requests:
            return {"total_requests": 0, "total_tokens": 0, "avg_tokens_per_sec": 0.0}
        total_tokens = sum(r["tokens"] for r in self._requests)
        total_time_s = sum(r["total_ms"] for r in self._requests) / 1000.0
        return {
            "total_requests": len(self._requests),
            "total_tokens": total_tokens,
            "avg_tokens_per_sec": round(total_tokens / total_time_s, 2) if total_time_s > 0 else 0.0,
            "total_time_s": round(total_time_s, 2),
        }
