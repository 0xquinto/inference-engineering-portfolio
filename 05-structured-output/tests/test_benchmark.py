import pytest

from src.benchmark import StructuredResult, StructuredBenchmarker


class TestStructuredResult:
    def test_tokens_per_sec(self):
        r = StructuredResult(
            backend_name="xgrammar",
            schema_name="simple_json",
            concurrency=10,
            ttft_ms=35.0,
            total_time_ms=500.0,
            tokens_generated=50,
            valid=True,
            retries=0,
        )
        assert r.tokens_per_sec == 100.0

    def test_tokens_per_sec_zero_time(self):
        r = StructuredResult(
            backend_name="xgrammar",
            schema_name="simple_json",
            concurrency=1,
            ttft_ms=0.0,
            total_time_ms=0.0,
            tokens_generated=50,
            valid=True,
            retries=0,
        )
        assert r.tokens_per_sec == 0.0

    def test_to_dict(self):
        r = StructuredResult(
            backend_name="unconstrained",
            schema_name="function_call",
            concurrency=1,
            ttft_ms=33.0,
            total_time_ms=1000.0,
            tokens_generated=100,
            valid=False,
            retries=2,
        )
        d = r.to_dict()
        assert d["backend_name"] == "unconstrained"
        assert d["tokens_per_sec"] == 100.0
        assert d["valid"] is False
        assert d["retries"] == 2


class TestStructuredBenchmarker:
    def test_init(self):
        b = StructuredBenchmarker(port=8010, max_tokens=512)
        assert b.base_url == "http://localhost:8010"
        assert b.max_tokens == 512
        assert b.disable_thinking is False

    def test_init_disable_thinking(self):
        b = StructuredBenchmarker(port=8010, disable_thinking=True)
        assert b.disable_thinking is True

    def test_aggregate_results(self):
        results = [
            StructuredResult("xgrammar", "simple_json", 1, 30.0, 1000.0, 100, True, 0),
            StructuredResult("xgrammar", "simple_json", 1, 40.0, 1200.0, 100, True, 0),
            StructuredResult("xgrammar", "simple_json", 1, 35.0, 1100.0, 100, False, 1),
        ]
        agg = StructuredBenchmarker.aggregate(results)
        assert agg["count"] == 3
        assert agg["ttft_p50"] == pytest.approx(35.0, abs=1.0)
        assert agg["validity_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert agg["avg_retries"] == pytest.approx(1 / 3, abs=0.01)

    def test_aggregate_empty(self):
        agg = StructuredBenchmarker.aggregate([])
        assert agg["count"] == 0
