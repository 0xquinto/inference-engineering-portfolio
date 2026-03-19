from pathlib import Path

import pandas as pd


def prepare_cost_data(cost_table: list, api_comparison: dict) -> pd.DataFrame:
    """Prepare combined cost data for self-hosted models and API providers."""
    rows = []
    for entry in cost_table:
        rows.append({
            "provider": entry["model"],
            "cost_per_million": entry["cost_per_million"],
            "type": "self-hosted",
        })
    for api_name, prices in api_comparison.items():
        rows.append({
            "provider": api_name,
            "cost_per_million": prices.get("output_per_million", 0),
            "type": "api",
        })
    return pd.DataFrame(rows)


def prepare_distribution_data(decisions: list[dict]) -> pd.DataFrame:
    """Prepare routing distribution data from cascade decisions."""
    counts: dict[str, int] = {}
    for d in decisions:
        model = d.get("final_model", d.get("routed_to", "unknown"))
        counts[model] = counts.get(model, 0) + 1
    rows = [{"model": k, "count": v} for k, v in counts.items()]
    return pd.DataFrame(rows)


def plot_cost_per_token(
    cost_table: list[dict],
    api_comparison: dict,
    output_dir: str = "results/",
) -> None:
    """Bar chart: x=model/API, y=$/million tokens, self-hosted vs API."""
    import matplotlib.pyplot as plt

    df = prepare_cost_data(cost_table, api_comparison)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3" if t == "self-hosted" else "#FF9800" for t in df["type"]]
    ax.bar(df["provider"], df["cost_per_million"], color=colors)
    ax.set_ylabel("Cost per Million Tokens ($)")
    ax.set_title("Self-Hosted vs API Cost Comparison")
    ax.tick_params(axis="x", rotation=30)

    from matplotlib.patches import Patch
    legend = [Patch(color="#2196F3", label="Self-Hosted"), Patch(color="#FF9800", label="API")]
    ax.legend(handles=legend)

    plt.tight_layout()
    plt.savefig(out / "cost_per_token.png", dpi=150)
    plt.close()


def plot_cascade_distribution(
    decisions: list[dict],
    output_dir: str = "results/",
) -> None:
    """Pie chart showing % routed to each tier."""
    import matplotlib.pyplot as plt

    df = prepare_distribution_data(decisions)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(df["count"], labels=df["model"], autopct="%1.1f%%", startangle=90)
    ax.set_title("Cascade Routing Distribution")

    plt.tight_layout()
    plt.savefig(out / "cascade_distribution.png", dpi=150)
    plt.close()


def plot_quality_vs_cost(
    results: list[dict],
    output_dir: str = "results/",
) -> None:
    """Scatter: x=$/M tokens, y=quality score, one point per tier."""
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    for r in results:
        ax.scatter(
            r.get("cost_per_million", 0),
            r.get("quality_score", 0),
            s=100,
            alpha=0.7,
        )
        ax.annotate(
            r.get("model", ""),
            (r.get("cost_per_million", 0), r.get("quality_score", 0)),
            textcoords="offset points",
            xytext=(10, 5),
        )

    ax.set_xlabel("Cost per Million Tokens ($)")
    ax.set_ylabel("Quality Score")
    ax.set_title("Quality vs Cost by Model Tier")

    plt.tight_layout()
    plt.savefig(out / "quality_vs_cost.png", dpi=150)
    plt.close()


def plot_utilization_breakeven(
    cost_table: list[dict],
    api_comparison: dict,
    output_dir: str = "results/",
) -> None:
    """Line chart: x=GPU utilization %, y=$/M tokens, show where self-hosted crosses API price."""
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    utilizations = np.linspace(0.05, 1.0, 20)

    fig, ax = plt.subplots(figsize=(12, 7))

    for entry in cost_table:
        tps = entry.get("tps", 0)
        gpu_cost = entry.get("gpu_cost_per_hour", 0)
        if tps <= 0:
            continue
        costs = [(gpu_cost / 3600) / (tps * u) * 1_000_000 for u in utilizations]
        ax.plot(utilizations * 100, costs, marker="o", label=f"{entry['model']} (self-hosted)")

    for api_name, prices in api_comparison.items():
        api_cost = prices.get("output_per_million", 0)
        ax.axhline(y=api_cost, linestyle="--", alpha=0.7, label=f"{api_name}")

    ax.set_xlabel("GPU Utilization (%)")
    ax.set_ylabel("Cost per Million Tokens ($)")
    ax.set_title("Self-Hosted Cost vs API at Different Utilization Levels")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out / "utilization_breakeven.png", dpi=150)
    plt.close()
