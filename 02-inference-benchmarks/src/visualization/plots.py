"""Visualization: latency box plots, throughput scaling curves, memory comparison."""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def prepare_latency_data(results: dict) -> pd.DataFrame:
    """Flatten benchmark results into a latency DataFrame."""
    rows = []
    for engine, entries in results.items():
        for entry in entries:
            row = {"engine": engine, "category": entry["category"], "concurrency": entry["concurrency"]}
            row.update(entry["latency"])
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_throughput_data(results: dict) -> pd.DataFrame:
    """Flatten benchmark results into a throughput DataFrame."""
    rows = []
    for engine, entries in results.items():
        for entry in entries:
            row = {"engine": engine, "category": entry["category"], "concurrency": entry["concurrency"]}
            row.update(entry["throughput"])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_latency_comparison(results: dict, output_dir: str = "results/"):
    """Generate latency comparison bar chart (P50/P95 by engine)."""
    df = prepare_latency_data(results)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for conc in df["concurrency"].unique():
        subset = df[df["concurrency"] == conc]
        fig, ax = plt.subplots(figsize=(10, 6))
        engines = subset["engine"].unique()
        x = range(len(engines))
        width = 0.35

        p50_vals = [subset[subset["engine"] == e]["p50"].mean() for e in engines]
        p95_vals = [subset[subset["engine"] == e]["p95"].mean() for e in engines]

        ax.bar([i - width/2 for i in x], p50_vals, width, label="P50", color="#2196F3")
        ax.bar([i + width/2 for i in x], p95_vals, width, label="P95", color="#FF9800")
        ax.set_xlabel("Engine")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Latency Comparison — Concurrency {conc}")
        ax.set_xticks(x)
        ax.set_xticklabels(engines)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_c{conc}.png", dpi=150)
        plt.close()


def plot_throughput_scaling(results: dict, output_dir: str = "results/"):
    """Generate throughput scaling curves (tok/s vs concurrency)."""
    df = prepare_throughput_data(results)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for engine in df["engine"].unique():
        subset = df[df["engine"] == engine].groupby("concurrency")["avg_tokens_per_sec"].mean()
        ax.plot(subset.index, subset.values, marker="o", label=engine, linewidth=2)

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Throughput Scaling: vLLM vs SGLang vs TensorRT-LLM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/throughput_scaling.png", dpi=150)
    plt.close()


def generate_all_plots(results_path: str, output_dir: str = "results/"):
    """Load results JSON and generate all plots."""
    with open(results_path) as f:
        results = json.load(f)
    plot_latency_comparison(results, output_dir)
    plot_throughput_scaling(results, output_dir)
    print(f"Charts saved to {output_dir}/")
