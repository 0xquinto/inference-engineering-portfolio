"""Latency tracking: TTFT, TPOT, and total latency percentiles."""

import statistics


class LatencyTracker:
    """Tracks per-request latency metrics and computes percentiles."""

    def __init__(self):
        self._total_ms: list[float] = []
        self._ttft_ms: list[float] = []
        self._tpot_ms: list[float] = []

    def record(self, total_ms: float):
        """Record a simple total latency measurement."""
        self._total_ms.append(total_ms)

    def record_request(self, ttft_ms: float, total_ms: float, tokens_generated: int):
        """Record a full request with TTFT and token-level metrics."""
        self._total_ms.append(total_ms)
        self._ttft_ms.append(ttft_ms)
        if tokens_generated > 0:
            tpot = (total_ms - ttft_ms) / tokens_generated
            self._tpot_ms.append(tpot)

    def summary(self) -> dict:
        result = {
            "count": len(self._total_ms),
            "p50": self._percentile(self._total_ms, 50),
            "p95": self._percentile(self._total_ms, 95),
            "p99": self._percentile(self._total_ms, 99),
            "mean": statistics.mean(self._total_ms) if self._total_ms else 0.0,
        }
        if self._ttft_ms:
            result["ttft_p50"] = self._percentile(self._ttft_ms, 50)
            result["ttft_p95"] = self._percentile(self._ttft_ms, 95)
        if self._tpot_ms:
            result["tpot_p50"] = self._percentile(self._tpot_ms, 50)
            result["tpot_p95"] = self._percentile(self._tpot_ms, 95)
        return result

    def _percentile(self, data: list[float], pct: int) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * pct / 100)
        idx = min(idx, len(sorted_data) - 1)
        return round(sorted_data[idx], 2)
