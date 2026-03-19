import pytest

from src.methods import MethodResult, SpecMethodTracker


class TestMethodResult:
    def test_tokens_per_sec(self):
        r = MethodResult(
            method_name="eagle3",
            qps=10,
            ttft_ms=38.0,
            total_time_ms=500.0,
            tokens_generated=50,
            acceptance_rate=0.85,
        )
        assert r.tokens_per_sec == 100.0

    def test_tokens_per_sec_zero_time(self):
        r = MethodResult(
            method_name="eagle3",
            qps=1,
            ttft_ms=0.0,
            total_time_ms=0.0,
            tokens_generated=50,
            acceptance_rate=0.85,
        )
        assert r.tokens_per_sec == 0.0

    def test_to_dict(self):
        r = MethodResult(
            method_name="baseline",
            qps=1,
            ttft_ms=42.0,
            total_time_ms=1000.0,
            tokens_generated=100,
            acceptance_rate=None,
        )
        d = r.to_dict()
        assert d["method_name"] == "baseline"
        assert d["tokens_per_sec"] == 100.0
        assert d["acceptance_rate"] is None

    def test_to_dict_with_acceptance(self):
        r = MethodResult(
            method_name="eagle3",
            qps=5,
            ttft_ms=38.0,
            total_time_ms=500.0,
            tokens_generated=85,
            acceptance_rate=0.82,
        )
        d = r.to_dict()
        assert d["acceptance_rate"] == 0.82


class TestSpecMethodTracker:
    def _make_tracker(self) -> SpecMethodTracker:
        tracker = SpecMethodTracker()
        # Baseline results at qps=1
        tracker.record(MethodResult("baseline", 1, 40.0, 1000.0, 100, None))
        tracker.record(MethodResult("baseline", 1, 50.0, 1200.0, 100, None))
        tracker.record(MethodResult("baseline", 1, 45.0, 1100.0, 100, None))
        # Eagle results at qps=1
        tracker.record(MethodResult("eagle3", 1, 35.0, 500.0, 100, 0.85))
        tracker.record(MethodResult("eagle3", 1, 38.0, 550.0, 100, 0.80))
        tracker.record(MethodResult("eagle3", 1, 36.0, 520.0, 100, 0.82))
        return tracker

    def test_summary_count(self):
        tracker = self._make_tracker()
        summary = tracker.summary("baseline", 1)
        assert summary["count"] == 3

    def test_summary_ttft(self):
        tracker = self._make_tracker()
        summary = tracker.summary("baseline", 1)
        assert summary["ttft_p50"] == pytest.approx(45.0, abs=1.0)

    def test_summary_throughput(self):
        tracker = self._make_tracker()
        summary = tracker.summary("eagle3", 1)
        assert summary["throughput_p50"] > 0

    def test_summary_acceptance_rate(self):
        tracker = self._make_tracker()
        summary = tracker.summary("eagle3", 1)
        assert summary["acceptance_rate_mean"] == pytest.approx(0.8233, abs=0.01)

    def test_summary_no_acceptance_for_baseline(self):
        tracker = self._make_tracker()
        summary = tracker.summary("baseline", 1)
        assert summary["acceptance_rate_mean"] is None

    def test_summary_empty(self):
        tracker = SpecMethodTracker()
        summary = tracker.summary("nonexistent", 1)
        assert summary["count"] == 0

    def test_speedup(self):
        tracker = self._make_tracker()
        speedup = tracker.speedup("eagle3", 1)
        assert speedup > 1.0

    def test_speedup_baseline_is_one(self):
        tracker = self._make_tracker()
        speedup = tracker.speedup("baseline", 1)
        assert speedup == pytest.approx(1.0, abs=0.01)

    def test_speedup_missing_method(self):
        tracker = self._make_tracker()
        speedup = tracker.speedup("nonexistent", 1)
        assert speedup == 0.0
