import pytest

from src.cost_model import (
    CostEstimate,
    calculate_cost_per_million_tokens,
    calculate_monthly_capacity,
    build_cost_table,
    compare_with_apis,
    cascade_cost_estimate,
)
from src.cascade import CascadeDecision
from src.config import ModelTier


class TestCalculateCostPerMillionTokens:
    def test_known_values(self):
        # 100 tps, $0.75/hr GPU
        # (0.75 / 3600) / 100 * 1_000_000 = 2.0833...
        result = calculate_cost_per_million_tokens(100.0, 0.75)
        assert result == pytest.approx(2.083, abs=0.01)

    def test_high_throughput_low_cost(self):
        result = calculate_cost_per_million_tokens(250.0, 0.75)
        assert result == pytest.approx(0.833, abs=0.01)

    def test_low_throughput_high_cost(self):
        result = calculate_cost_per_million_tokens(25.0, 3.59)
        assert result == pytest.approx(39.89, abs=0.1)

    def test_zero_tps_returns_inf(self):
        result = calculate_cost_per_million_tokens(0.0, 0.75)
        assert result == float("inf")


class TestCalculateMonthlyCapacity:
    def test_basic(self):
        # 100 tps * 3600 * 720 * 0.5 = 129,600,000
        result = calculate_monthly_capacity(100.0, 0.5)
        assert result == pytest.approx(129_600_000, rel=0.01)

    def test_full_utilization(self):
        result = calculate_monthly_capacity(100.0, 1.0)
        assert result == pytest.approx(259_200_000, rel=0.01)

    def test_zero_tps(self):
        result = calculate_monthly_capacity(0.0, 0.5)
        assert result == 0.0


class TestBuildCostTable:
    def test_builds_table(self):
        models = [
            ModelTier("small", "model-a", "0.5B", "Small", 8010, 0.75, 1200),
            ModelTier("large", "model-b", "72B", "Large", 8012, 3.59, 45000),
        ]
        tps = {"small": 250.0, "large": 25.0}
        table = build_cost_table(models, tps, 0.5)

        assert len(table) == 2
        assert all(isinstance(e, CostEstimate) for e in table)
        assert table[0].model_name == "small"
        assert table[0].tokens_per_second == 250.0
        assert table[0].cost_per_million_tokens == pytest.approx(0.833, abs=0.01)

    def test_missing_tps_defaults_to_zero(self):
        models = [ModelTier("unknown", "model-x", "1B", "Unknown", 8010, 1.0, 2000)]
        table = build_cost_table(models, {}, 0.5)
        assert table[0].cost_per_million_tokens == float("inf")


class TestCompareWithApis:
    def test_returns_breakeven_data(self):
        estimates = [
            CostEstimate("small", 250.0, 0.75, 0.83, 540.0, 648_000_000),
        ]
        api_prices = {
            "openai_gpt4o": {"input_per_million": 2.50, "output_per_million": 10.00},
        }
        result = compare_with_apis(estimates, api_prices)

        assert "small" in result
        assert "openai_gpt4o" in result["small"]
        cmp = result["small"]["openai_gpt4o"]
        assert cmp["self_hosted_cost_per_m"] == 0.83
        assert cmp["api_cost_per_m"] == 10.00
        assert cmp["savings_pct"] > 0  # self-hosted is cheaper

    def test_multiple_apis(self):
        estimates = [
            CostEstimate("medium", 85.0, 0.75, 2.45, 540.0, 220_320_000),
        ]
        api_prices = {
            "openai_gpt4o": {"output_per_million": 10.00},
            "openai_gpt4o_mini": {"output_per_million": 0.60},
        }
        result = compare_with_apis(estimates, api_prices)
        assert "openai_gpt4o" in result["medium"]
        assert "openai_gpt4o_mini" in result["medium"]
        # Should be cheaper than GPT-4o but more expensive than mini
        assert result["medium"]["openai_gpt4o"]["savings_pct"] > 0
        assert result["medium"]["openai_gpt4o_mini"]["savings_pct"] < 0


class TestCascadeCostEstimate:
    def test_weighted_cost(self):
        decisions = [
            CascadeDecision("q1", "simple", "small", False, "small"),
            CascadeDecision("q2", "simple", "small", False, "small"),
            CascadeDecision("q3", "moderate", "medium", False, "medium"),
            CascadeDecision("q4", "complex", "large", False, "large"),
        ]
        estimates = [
            CostEstimate("small", 250.0, 0.75, 1.0, 540.0, 648_000_000),
            CostEstimate("medium", 85.0, 0.75, 3.0, 540.0, 220_320_000),
            CostEstimate("large", 25.0, 3.59, 40.0, 2584.80, 64_800_000),
        ]
        result = cascade_cost_estimate(decisions, estimates)

        assert "distribution" in result
        assert result["distribution"]["small"]["count"] == 2
        assert result["distribution"]["small"]["percentage"] == pytest.approx(50.0)
        assert result["blended_cost_per_million"] > 0
        # Blended: 0.5*1.0 + 0.25*3.0 + 0.25*40.0 = 0.5 + 0.75 + 10.0 = 11.25
        assert result["blended_cost_per_million"] == pytest.approx(11.25)

    def test_empty_decisions(self):
        result = cascade_cost_estimate([], [])
        assert result["blended_cost_per_million"] == 0.0
        assert result["distribution"] == {}
