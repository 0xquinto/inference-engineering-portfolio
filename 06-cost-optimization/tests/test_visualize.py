import pytest

from src.visualize import (
    prepare_cost_data,
    prepare_distribution_data,
    plot_cost_per_token,
    plot_cascade_distribution,
    plot_quality_vs_cost,
    plot_utilization_breakeven,
)


class TestPrepareCostData:
    def test_basic(self, sample_cost_table):
        api_cmp = {
            "openai_gpt4o": {"output_per_million": 10.00},
            "openai_gpt4o_mini": {"output_per_million": 0.60},
        }
        df = prepare_cost_data(sample_cost_table, api_cmp)
        assert len(df) == 5  # 3 self-hosted + 2 API
        assert "provider" in df.columns
        assert "cost_per_million" in df.columns
        assert "type" in df.columns
        assert set(df["type"].unique()) == {"self-hosted", "api"}


class TestPrepareDistributionData:
    def test_basic(self):
        decisions = [
            {"final_model": "small"},
            {"final_model": "small"},
            {"final_model": "medium"},
            {"final_model": "large"},
        ]
        df = prepare_distribution_data(decisions)
        assert len(df) == 3
        assert "model" in df.columns
        assert "count" in df.columns
        assert df[df["model"] == "small"]["count"].values[0] == 2


class TestPlotCostPerToken:
    def test_no_crash(self, sample_cost_table, tmp_path):
        api_cmp = {"openai_gpt4o": {"output_per_million": 10.00}}
        plot_cost_per_token(sample_cost_table, api_cmp, str(tmp_path))
        assert (tmp_path / "cost_per_token.png").exists()


class TestPlotCascadeDistribution:
    def test_no_crash(self, tmp_path):
        decisions = [
            {"final_model": "small"},
            {"final_model": "small"},
            {"final_model": "medium"},
            {"final_model": "large"},
        ]
        plot_cascade_distribution(decisions, str(tmp_path))
        assert (tmp_path / "cascade_distribution.png").exists()


class TestPlotQualityVsCost:
    def test_no_crash(self, tmp_path):
        results = [
            {"model": "small", "cost_per_million": 0.83, "quality_score": 0.7},
            {"model": "medium", "cost_per_million": 2.45, "quality_score": 0.85},
            {"model": "large", "cost_per_million": 39.89, "quality_score": 0.95},
        ]
        plot_quality_vs_cost(results, str(tmp_path))
        assert (tmp_path / "quality_vs_cost.png").exists()


class TestPlotUtilizationBreakeven:
    def test_no_crash(self, sample_cost_table, tmp_path):
        api_cmp = {
            "openai_gpt4o": {"output_per_million": 10.00},
            "openai_gpt4o_mini": {"output_per_million": 0.60},
        }
        plot_utilization_breakeven(sample_cost_table, api_cmp, str(tmp_path))
        assert (tmp_path / "utilization_breakeven.png").exists()
