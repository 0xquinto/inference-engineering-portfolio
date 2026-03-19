from pathlib import Path

import pandas as pd


def prepare_ttft_data(results: dict) -> pd.DataFrame:
    rows = []
    for method_name, qps_data in results.items():
        for qps_str, stats in qps_data.items():
            rows.append({
                "method": method_name,
                "qps": int(qps_str),
                "ttft_p50": stats.get("ttft_p50", 0),
                "throughput_p50": stats.get("throughput_p50", 0),
            })
    return pd.DataFrame(rows)


def prepare_speedup_data(results: dict, methods: list[str], qps_levels: list[int]) -> pd.DataFrame:
    rows = []
    for method in methods:
        row = {"method": method}
        for qps in qps_levels:
            qps_str = str(qps)
            baseline_tps = results.get("baseline", {}).get(qps_str, {}).get("throughput_p50", 0)
            method_tps = results.get(method, {}).get(qps_str, {}).get("throughput_p50", 0)
            if baseline_tps > 0:
                row[f"qps_{qps}"] = method_tps / baseline_tps
            else:
                row[f"qps_{qps}"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_ttft_by_qps(results: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt

    df = prepare_ttft_data(results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in df["method"].unique():
        method_df = df[df["method"] == method].sort_values("qps")
        ax.plot(method_df["qps"], method_df["ttft_p50"], marker="o", label=method)

    ax.set_xlabel("QPS")
    ax.set_ylabel("TTFT p50 (ms)")
    ax.set_title("Time to First Token by QPS Level")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "ttft_by_qps.png", dpi=150)
    plt.close()


def plot_throughput_by_qps(results: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt

    df = prepare_ttft_data(results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in df["method"].unique():
        method_df = df[df["method"] == method].sort_values("qps")
        ax.plot(method_df["qps"], method_df["throughput_p50"], marker="o", label=method)

    ax.set_xlabel("QPS")
    ax.set_ylabel("Throughput p50 (tokens/sec)")
    ax.set_title("Throughput by QPS Level")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "throughput_by_qps.png", dpi=150)
    plt.close()


def plot_speedup_heatmap(results: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    methods = [m for m in results.keys() if m != "baseline"]
    if not methods:
        return

    qps_levels = sorted({int(q) for m_data in results.values() for q in m_data.keys()})
    if not qps_levels:
        return

    df = prepare_speedup_data(results, methods, qps_levels)

    data = []
    for _, row in df.iterrows():
        data.append([row.get(f"qps_{q}", 0.0) for q in qps_levels])
    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, max(4, len(methods) * 0.8 + 2)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=2.5)

    ax.set_xticks(range(len(qps_levels)))
    ax.set_xticklabels([str(q) for q in qps_levels])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("QPS")
    ax.set_ylabel("Method")
    ax.set_title("Speedup vs Baseline")

    for i in range(len(methods)):
        for j in range(len(qps_levels)):
            ax.text(j, i, f"{data[i, j]:.2f}x", ha="center", va="center", fontsize=10)

    fig.colorbar(im, ax=ax, label="Speedup")
    plt.tight_layout()
    plt.savefig(out / "speedup_heatmap.png", dpi=150)
    plt.close()
