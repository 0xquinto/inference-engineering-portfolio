from dataclasses import dataclass

from .cascade import CascadeDecision
from .config import ModelTier


@dataclass
class CostEstimate:
    model_name: str
    tokens_per_second: float
    gpu_cost_per_hour: float
    cost_per_million_tokens: float
    monthly_cost_at_utilization: float
    monthly_token_capacity: float


def calculate_cost_per_million_tokens(tps: float, gpu_cost_per_hour: float) -> float:
    """Cost per million output tokens for self-hosted inference.

    Formula: (gpu_cost_per_hour / 3600) / tps * 1_000_000
    """
    if tps <= 0:
        return float("inf")
    return (gpu_cost_per_hour / 3600) / tps * 1_000_000


def calculate_monthly_capacity(tps: float, utilization: float) -> float:
    """Total tokens a model can produce in a month at given utilization.

    Formula: tps * 3600 * 720 * utilization
    """
    return tps * 3600 * 720 * utilization


def build_cost_table(
    models: list[ModelTier],
    tps_by_model: dict[str, float],
    utilization: float,
) -> list[CostEstimate]:
    """Build a cost table for all model tiers given measured tokens/sec."""
    table = []
    for model in models:
        tps = tps_by_model.get(model.name, 0.0)
        cost_per_m = calculate_cost_per_million_tokens(tps, model.gpu_cost_per_hour)
        monthly_cap = calculate_monthly_capacity(tps, utilization)
        monthly_cost = model.gpu_cost_per_hour * 720  # full month of GPU time
        table.append(CostEstimate(
            model_name=model.name,
            tokens_per_second=tps,
            gpu_cost_per_hour=model.gpu_cost_per_hour,
            cost_per_million_tokens=cost_per_m,
            monthly_cost_at_utilization=monthly_cost,
            monthly_token_capacity=monthly_cap,
        ))
    return table


def compare_with_apis(
    self_hosted: list[CostEstimate],
    api_prices: dict,
) -> dict:
    """Compare self-hosted cost per million tokens with API prices.

    Returns a dict mapping each self-hosted model to breakeven info
    against each API provider.
    """
    comparisons = {}
    for estimate in self_hosted:
        model_cmp = {}
        for api_name, prices in api_prices.items():
            api_output_cost = prices.get("output_per_million", 0)
            savings_pct = 0.0
            if api_output_cost > 0:
                savings_pct = (1 - estimate.cost_per_million_tokens / api_output_cost) * 100
            model_cmp[api_name] = {
                "self_hosted_cost_per_m": estimate.cost_per_million_tokens,
                "api_cost_per_m": api_output_cost,
                "savings_pct": savings_pct,
            }
        comparisons[estimate.model_name] = model_cmp
    return comparisons


def cascade_cost_estimate(
    decisions: list[CascadeDecision],
    cost_estimates: list[CostEstimate],
) -> dict:
    """Compute weighted cost based on routing distribution.

    Returns distribution percentages and blended cost per million tokens.
    """
    if not decisions:
        return {"distribution": {}, "blended_cost_per_million": 0.0}

    cost_lookup = {e.model_name: e for e in cost_estimates}
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.final_model] = counts.get(d.final_model, 0) + 1

    total = len(decisions)
    distribution = {}
    blended_cost = 0.0

    for model_name, count in counts.items():
        pct = count / total
        distribution[model_name] = {
            "count": count,
            "percentage": pct * 100,
        }
        estimate = cost_lookup.get(model_name)
        if estimate:
            blended_cost += pct * estimate.cost_per_million_tokens

    return {
        "distribution": distribution,
        "blended_cost_per_million": blended_cost,
    }
