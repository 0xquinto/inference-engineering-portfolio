"""
Metrics collection and cost tracking.

Tracks per-request metrics and provides a summary endpoint
showing routing decisions, cost savings, and latency stats.
"""

import time
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


@dataclass
class RequestMetric:
    timestamp: float
    model_key: str           # "small" or "large"
    complexity_score: float
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    total_time_ms: float
    had_tool_calls: bool
    cost_usd: float


class MetricsCollector:
    """In-memory metrics store. Replace with Prometheus/InfluxDB for production."""

    def __init__(self):
        self._metrics: list[RequestMetric] = []

    def record(self, metric: RequestMetric):
        self._metrics.append(metric)

    def summary(self) -> dict:
        if not self._metrics:
            return {"total_requests": 0}

        total = len(self._metrics)
        by_model = defaultdict(list)
        for m in self._metrics:
            by_model[m.model_key].append(m)

        # Latency stats
        all_latencies = [m.total_time_ms for m in self._metrics]
        small_latencies = [m.total_time_ms for m in by_model.get("small", [])]
        large_latencies = [m.total_time_ms for m in by_model.get("large", [])]

        # Cost analysis
        total_cost = sum(m.cost_usd for m in self._metrics)
        # What would it cost if everything went to the large model?
        cost_if_all_large = sum(
            self._estimate_large_cost(m.input_tokens, m.output_tokens)
            for m in self._metrics
        )
        cost_savings = cost_if_all_large - total_cost

        return {
            "total_requests": total,
            "routing_split": {
                "small": len(by_model.get("small", [])),
                "large": len(by_model.get("large", [])),
                "small_pct": len(by_model.get("small", [])) / total * 100,
            },
            "latency_ms": {
                "p50": self._percentile(all_latencies, 50),
                "p95": self._percentile(all_latencies, 95),
                "p99": self._percentile(all_latencies, 99),
                "small_avg": statistics.mean(small_latencies) if small_latencies else 0,
                "large_avg": statistics.mean(large_latencies) if large_latencies else 0,
            },
            "cost": {
                "total_usd": round(total_cost, 6),
                "if_all_large_usd": round(cost_if_all_large, 6),
                "savings_usd": round(cost_savings, 6),
                "savings_pct": round(cost_savings / cost_if_all_large * 100, 1) if cost_if_all_large > 0 else 0,
            },
            "tokens": {
                "total_input": sum(m.input_tokens for m in self._metrics),
                "total_output": sum(m.output_tokens for m in self._metrics),
            },
            "tool_calls": {
                "total": sum(1 for m in self._metrics if m.had_tool_calls),
                "pct": sum(1 for m in self._metrics if m.had_tool_calls) / total * 100,
            },
        }

    def _estimate_large_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate what this request would cost on the large model."""
        return (input_tokens * 0.80 + output_tokens * 0.80) / 1_000_000

    def _percentile(self, data: list[float], pct: int) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * pct / 100)
        idx = min(idx, len(sorted_data) - 1)
        return round(sorted_data[idx], 2)
