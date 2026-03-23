import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_goodput_vs_qps(data: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, qps_data in data.items():
        qps_vals = sorted(int(q) for q in qps_data.keys())
        goodput_vals = [qps_data[str(q) if str(q) in qps_data else q]["goodput"] * 100 for q in qps_vals]
        ax.plot(qps_vals, goodput_vals, marker="o", label=policy.upper())
    ax.set_xlabel("QPS (Queries Per Second)")
    ax.set_ylabel("Goodput (% meeting SLO)")
    ax.set_title("Goodput vs Load by Scheduling Policy")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "goodput_vs_qps.png"), dpi=150)
    plt.close(fig)


def plot_latency_cdf(data: dict, slos: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, qps_data in data.items():
        for qps, metrics in qps_data.items():
            latencies = sorted(metrics.get("latencies", []))
            if not latencies:
                continue
            cdf = np.arange(1, len(latencies) + 1) / len(latencies)
            ax.plot(latencies, cdf, label=f"{policy.upper()} QPS={qps}")
    for cls, slo in slos.items():
        ax.axvline(x=slo, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("End-to-End Latency (seconds)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency Distribution by Policy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "latency_cdf.png"), dpi=150)
    plt.close(fig)


def plot_fairness_heatmap(data: dict, output_dir: str):
    policies = sorted(data.keys())
    qps_vals = sorted({q for p in data.values() for q in p.keys()})
    matrix = []
    for policy in policies:
        row = [data[policy].get(q, {}).get("fairness", 0) for q in qps_vals]
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(qps_vals)))
    ax.set_xticklabels([str(q) for q in qps_vals])
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels([p.upper() for p in policies])
    ax.set_xlabel("QPS")
    ax.set_title("Fairness (min/max class goodput)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fairness_heatmap.png"), dpi=150)
    plt.close(fig)
