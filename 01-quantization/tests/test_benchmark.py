import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.benchmark import BenchmarkResult, PerfBenchmarker


class TestBenchmarkResult:
    def test_tokens_per_sec(self):
        r = BenchmarkResult(
            format_name="gptq_int4",
            concurrency=10,
            ttft_ms=32.1,
            total_time_ms=500.0,
            tokens_generated=50,
            gpu_memory_mb=5100,
        )
        assert r.tokens_per_sec == 100.0

    def test_tokens_per_sec_zero_time(self):
        r = BenchmarkResult(
            format_name="gptq_int4",
            concurrency=1,
            ttft_ms=0.0,
            total_time_ms=0.0,
            tokens_generated=50,
            gpu_memory_mb=5100,
        )
        assert r.tokens_per_sec == 0.0

    def test_to_dict(self):
        r = BenchmarkResult(
            format_name="bf16",
            concurrency=1,
            ttft_ms=45.0,
            total_time_ms=1000.0,
            tokens_generated=100,
            gpu_memory_mb=14200,
        )
        d = r.to_dict()
        assert d["format_name"] == "bf16"
        assert d["tokens_per_sec"] == 100.0


class TestPerfBenchmarker:
    def test_init(self):
        b = PerfBenchmarker(port=8010, max_tokens=256)
        assert b.base_url == "http://localhost:8010"

    def test_aggregate_results(self):
        results = [
            BenchmarkResult("bf16", 1, 40.0, 1000.0, 100, 14000),
            BenchmarkResult("bf16", 1, 50.0, 1200.0, 100, 14000),
            BenchmarkResult("bf16", 1, 45.0, 1100.0, 100, 14000),
        ]
        agg = PerfBenchmarker.aggregate(results)
        assert agg["count"] == 3
        assert agg["ttft_p50"] == pytest.approx(45.0, abs=1.0)
        assert agg["throughput_p50"] == pytest.approx(100.0, abs=10.0)
