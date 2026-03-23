import pytest

from src.metrics import RequestResult, compute_goodput, compute_goodput_per_class, compute_fairness, compute_latency_percentiles


class TestRequestResult:
    def test_meets_slo(self):
        r = RequestResult("short", slo_seconds=2.0, latency_seconds=1.5,
                          ttft_seconds=0.1, tokens=50)
        assert r.meets_slo is True

    def test_violates_slo(self):
        r = RequestResult("short", slo_seconds=2.0, latency_seconds=3.0,
                          ttft_seconds=0.1, tokens=50)
        assert r.meets_slo is False

    def test_exact_slo_meets(self):
        r = RequestResult("short", slo_seconds=2.0, latency_seconds=2.0,
                          ttft_seconds=0.1, tokens=50)
        assert r.meets_slo is True


class TestComputeGoodput:
    def test_all_pass(self):
        results = [
            RequestResult("short", 2.0, 1.0, 0.1, 50),
            RequestResult("short", 2.0, 1.5, 0.1, 50),
        ]
        assert compute_goodput(results) == 1.0

    def test_all_fail(self):
        results = [
            RequestResult("short", 2.0, 3.0, 0.1, 50),
            RequestResult("short", 2.0, 5.0, 0.1, 50),
        ]
        assert compute_goodput(results) == 0.0

    def test_partial(self):
        results = [
            RequestResult("short", 2.0, 1.0, 0.1, 50),
            RequestResult("short", 2.0, 3.0, 0.1, 50),
            RequestResult("medium", 8.0, 5.0, 0.1, 200),
            RequestResult("medium", 8.0, 10.0, 0.1, 200),
        ]
        assert compute_goodput(results) == 0.5

    def test_empty(self):
        assert compute_goodput([]) == 0.0

    def test_per_class(self):
        results = [
            RequestResult("short", 2.0, 1.0, 0.1, 50),
            RequestResult("short", 2.0, 3.0, 0.1, 50),
            RequestResult("long", 20.0, 10.0, 0.1, 500),
        ]
        per_class = compute_goodput_per_class(results)
        assert per_class["short"] == 0.5
        assert per_class["long"] == 1.0


class TestComputeFairness:
    def test_perfect_fairness(self):
        per_class = {"short": 1.0, "medium": 1.0, "long": 1.0}
        assert compute_fairness(per_class) == 1.0

    def test_unfair(self):
        per_class = {"short": 1.0, "medium": 0.5, "long": 0.0}
        assert compute_fairness(per_class) == 0.0

    def test_empty(self):
        assert compute_fairness({}) == 0.0


class TestComputeLatencyPercentiles:
    def test_basic(self):
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        p = compute_latency_percentiles(latencies)
        assert p["p50"] == pytest.approx(5.5, abs=0.5)
        assert p["p95"] == pytest.approx(10.0, abs=0.5)
        assert p["p99"] == pytest.approx(10.0, abs=0.5)

    def test_empty(self):
        p = compute_latency_percentiles([])
        assert p["p50"] == 0.0
