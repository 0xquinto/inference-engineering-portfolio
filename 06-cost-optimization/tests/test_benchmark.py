import pytest

from src.benchmark import CascadeResult, CascadeBenchmarker


class TestCascadeResult:
    def test_tokens_per_sec(self):
        r = CascadeResult(
            model_name="small",
            complexity="simple",
            prompt="What is AI?",
            ttft_ms=20.0,
            total_time_ms=500.0,
            tokens_generated=50,
            quality_score=0.9,
            escalated=False,
        )
        assert r.tokens_per_sec == 100.0

    def test_tokens_per_sec_zero_time(self):
        r = CascadeResult(
            model_name="small",
            complexity="simple",
            prompt="What is AI?",
            ttft_ms=0.0,
            total_time_ms=0.0,
            tokens_generated=50,
            quality_score=0.9,
            escalated=False,
        )
        assert r.tokens_per_sec == 0.0

    def test_to_dict(self):
        r = CascadeResult(
            model_name="medium",
            complexity="moderate",
            prompt="Explain TCP.",
            ttft_ms=30.0,
            total_time_ms=1000.0,
            tokens_generated=100,
            quality_score=0.85,
            escalated=False,
        )
        d = r.to_dict()
        assert d["model_name"] == "medium"
        assert d["complexity"] == "moderate"
        assert d["tokens_per_sec"] == 100.0
        assert d["quality_score"] == 0.85


class TestCascadeBenchmarker:
    def test_init(self):
        b = CascadeBenchmarker(
            models={"small": 8010, "medium": 8011, "large": 8012},
            max_tokens=256,
        )
        assert b.models["small"] == 8010
        assert b.models["large"] == 8012
        assert b.max_tokens == 256

    def test_aggregate_results(self):
        results = [
            CascadeResult("small", "simple", "q1", 20.0, 500.0, 50, 0.9, False),
            CascadeResult("small", "simple", "q2", 25.0, 600.0, 60, 0.85, False),
            CascadeResult("medium", "moderate", "q3", 40.0, 1000.0, 100, 0.8, False),
        ]
        agg = CascadeBenchmarker.aggregate(results)
        assert agg["count"] == 3
        assert "ttft_p50" in agg
        assert "throughput_p50" in agg
        assert "per_model" in agg
        assert "small" in agg["per_model"]
        assert agg["per_model"]["small"]["count"] == 2
        assert "medium" in agg["per_model"]
        assert agg["per_model"]["medium"]["count"] == 1

    def test_aggregate_empty(self):
        agg = CascadeBenchmarker.aggregate([])
        assert agg["count"] == 0
