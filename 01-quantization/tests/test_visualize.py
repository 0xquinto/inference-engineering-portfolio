import pytest

from src.visualize import prepare_comparison_data, prepare_pareto_data


class TestPrepareComparisonData:
    def test_basic(self, sample_results):
        df = prepare_comparison_data(sample_results)
        assert len(df) == 2
        assert "format" in df.columns
        assert "vram_mb" in df.columns

    def test_columns(self, sample_results):
        df = prepare_comparison_data(sample_results)
        expected_cols = {"format", "perplexity", "mmlu_accuracy", "vram_mb", "throughput_c1", "ttft_c1"}
        assert expected_cols.issubset(set(df.columns))


class TestPreparePareto:
    def test_pareto_points(self, sample_results):
        points = prepare_pareto_data(sample_results)
        assert len(points) == 2
        assert all("format" in p for p in points)
        assert all("quality" in p for p in points)
        assert all("speed" in p for p in points)
        assert all("memory" in p for p in points)
