import pytest

from src.visualize import prepare_ttft_comparison, prepare_cache_pressure_curve


class TestPrepareTTFTComparison:
    def test_basic(self):
        data = {
            "shared_system_prompt": {
                "caching_on": {"ttft_p50": 8.0, "count": 100},
                "caching_off": {"ttft_p50": 35.0, "count": 100},
            }
        }
        df = prepare_ttft_comparison(data)
        assert len(df) == 1
        assert df.iloc[0]["speedup"] == pytest.approx(4.375, abs=0.1)


class TestPrepareCachePressureCurve:
    def test_basic(self):
        data = {
            10: {"ttft_p50": 8.0},
            50: {"ttft_p50": 12.0},
            200: {"ttft_p50": 30.0},
        }
        points = prepare_cache_pressure_curve(data)
        assert len(points) == 3
        assert points[0]["unique_prefixes"] == 10
        assert points[2]["ttft_p50"] == 30.0
