import pytest

from src.metrics import CacheMetrics, RequestMetric


class TestRequestMetric:
    def test_to_dict(self):
        m = RequestMetric(scenario="shared_system", caching=True, request_idx=0, ttft_ms=12.3, total_ms=500.0, tokens=50)
        d = m.to_dict()
        assert d["ttft_ms"] == 12.3
        assert d["tokens_per_sec"] == 100.0

    def test_tps_zero_time(self):
        m = RequestMetric(scenario="test", caching=True, request_idx=0, ttft_ms=0, total_ms=0, tokens=50)
        assert m.tokens_per_sec == 0.0


class TestCacheMetrics:
    def test_record_and_summary(self):
        cm = CacheMetrics()
        cm.record(RequestMetric("s1", True, 0, 10.0, 500.0, 50))
        cm.record(RequestMetric("s1", True, 1, 12.0, 520.0, 50))
        cm.record(RequestMetric("s1", True, 2, 11.0, 510.0, 50))

        summary = cm.summary("s1", caching=True)
        assert summary["count"] == 3
        assert summary["ttft_p50"] == pytest.approx(11.0, abs=1.0)

    def test_separate_caching_on_off(self):
        cm = CacheMetrics()
        cm.record(RequestMetric("s1", True, 0, 10.0, 500.0, 50))
        cm.record(RequestMetric("s1", False, 0, 40.0, 600.0, 50))

        on = cm.summary("s1", caching=True)
        off = cm.summary("s1", caching=False)
        assert on["ttft_p50"] < off["ttft_p50"]

    def test_speedup(self):
        cm = CacheMetrics()
        for i in range(5):
            cm.record(RequestMetric("s1", False, i, 40.0, 600.0, 50))
            cm.record(RequestMetric("s1", True, i, 10.0, 500.0, 50))

        speedup = cm.ttft_speedup("s1")
        assert speedup == pytest.approx(4.0, abs=0.5)

    def test_empty_summary(self):
        cm = CacheMetrics()
        summary = cm.summary("nonexistent", caching=True)
        assert summary["count"] == 0
