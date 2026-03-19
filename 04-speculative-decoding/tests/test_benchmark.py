import pytest

from src.benchmark import SpecBenchmarker
from src.methods import MethodResult


class TestSpecBenchmarker:
    def test_init(self):
        b = SpecBenchmarker(port=8010, max_tokens=256)
        assert b.base_url == "http://localhost:8010"
        assert b.max_tokens == 256

    def test_init_custom_port(self):
        b = SpecBenchmarker(port=9090, max_tokens=128, temperature=0.5)
        assert b.base_url == "http://localhost:9090"
        assert b.max_tokens == 128
        assert b.temperature == 0.5

    def test_aggregate_results(self):
        results = [
            MethodResult("baseline", 1, 40.0, 1000.0, 100, None),
            MethodResult("baseline", 1, 50.0, 1200.0, 100, None),
            MethodResult("baseline", 1, 45.0, 1100.0, 100, None),
        ]
        agg = SpecBenchmarker.aggregate(results)
        assert agg["count"] == 3
        assert agg["ttft_p50"] == pytest.approx(45.0, abs=1.0)
        assert agg["throughput_p50"] == pytest.approx(100.0, abs=10.0)

    def test_aggregate_empty(self):
        agg = SpecBenchmarker.aggregate([])
        assert agg["count"] == 0
