"""Tests for metrics collection utilities."""

import pytest
from src.metrics.latency import LatencyTracker
from src.metrics.throughput import ThroughputCalculator
from src.metrics.memory import parse_gpu_memory


class TestLatencyTracker:
    def test_record_and_percentiles(self):
        tracker = LatencyTracker()
        for ms in [100, 200, 300, 400, 500]:
            tracker.record(ms)
        stats = tracker.summary()
        assert stats["count"] == 5
        assert stats["p50"] == 300.0
        assert stats["p95"] >= 400.0
        assert stats["p99"] >= 400.0
        assert stats["mean"] == 300.0

    def test_empty_tracker(self):
        tracker = LatencyTracker()
        stats = tracker.summary()
        assert stats["count"] == 0
        assert stats["p50"] == 0.0

    def test_ttft_and_tpot(self):
        tracker = LatencyTracker()
        tracker.record_request(ttft_ms=50.0, total_ms=500.0, tokens_generated=100)
        stats = tracker.summary()
        assert stats["count"] == 1
        assert stats["ttft_p50"] == 50.0
        assert stats["tpot_p50"] == pytest.approx(4.5, abs=0.1)  # (500-50)/100


class TestThroughputCalculator:
    def test_calculate_throughput(self):
        calc = ThroughputCalculator()
        calc.record_request(total_ms=1000.0, tokens_generated=100)
        calc.record_request(total_ms=2000.0, tokens_generated=200)
        stats = calc.summary()
        assert stats["total_requests"] == 2
        assert stats["total_tokens"] == 300
        assert stats["avg_tokens_per_sec"] > 0

    def test_empty_calculator(self):
        calc = ThroughputCalculator()
        stats = calc.summary()
        assert stats["total_requests"] == 0


class TestGpuMemory:
    def test_parse_gpu_memory_mock(self):
        # Test with mock nvidia-smi output
        mock_output = "45000 MiB, 81920 MiB"
        used, total = parse_gpu_memory(mock_output)
        assert used == 45000
        assert total == 81920
