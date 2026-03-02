from pathlib import Path

import pandas as pd


def prepare_ttft_comparison(scenario_summaries: dict) -> pd.DataFrame:
    rows = []
    for scenario, data in scenario_summaries.items():
        on = data.get("caching_on", {})
        off = data.get("caching_off", {})
        ttft_on = on.get("ttft_p50", 0)
        ttft_off = off.get("ttft_p50", 0)
        speedup = ttft_off / ttft_on if ttft_on > 0 else 0
        rows.append({
            "scenario": scenario,
            "ttft_caching_on": ttft_on,
            "ttft_caching_off": ttft_off,
            "speedup": speedup,
        })
    return pd.DataFrame(rows)


def prepare_cache_pressure_curve(pressure_data: dict) -> list[dict]:
    points = []
    for n_prefixes, summary in sorted(pressure_data.items()):
        points.append({
            "unique_prefixes": n_prefixes,
            "ttft_p50": summary.get("ttft_p50", 0),
        })
    return points


def plot_ttft_comparison(scenario_summaries: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt

    df = prepare_ttft_comparison(scenario_summaries)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = range(len(df))
    width = 0.35
    ax1.bar([i - width/2 for i in x], df["ttft_caching_off"], width, label="Caching OFF", color="#e74c3c")
    ax1.bar([i + width/2 for i in x], df["ttft_caching_on"], width, label="Caching ON", color="#2ecc71")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["scenario"], rotation=15, ha="right")
    ax1.set_ylabel("TTFT p50 (ms)")
    ax1.set_title("Time to First Token: Caching ON vs OFF")
    ax1.legend()

    ax2.barh(df["scenario"], df["speedup"], color="#3498db")
    ax2.set_xlabel("TTFT Speedup (x)")
    ax2.set_title("Prefix Caching Speedup by Scenario")

    plt.tight_layout()
    plt.savefig(out / "ttft_comparison.png", dpi=150)
    plt.close()


def plot_cache_pressure(pressure_data: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt

    points = prepare_cache_pressure_curve(pressure_data)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = [p["unique_prefixes"] for p in points]
    y = [p["ttft_p50"] for p in points]
    ax.plot(x, y, marker="o", linewidth=2, color="#e74c3c")
    ax.set_xlabel("Number of Unique Prefixes in Cache")
    ax.set_ylabel("TTFT p50 (ms)")
    ax.set_title("Cache Pressure: TTFT vs Unique Prefix Count")
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(out / "cache_pressure.png", dpi=150)
    plt.close()
