from pathlib import Path

import pandas as pd


def prepare_comparison_data(results: dict) -> pd.DataFrame:
    rows = []
    for fmt_name, data in results.items():
        row = {
            "format": fmt_name,
            "perplexity": data.get("perplexity", 0),
            "mmlu_accuracy": data.get("mmlu_accuracy", 0),
            "vram_mb": data.get("vram_mb", 0),
            "load_time_s": data.get("load_time_s", 0),
        }
        for level in ["1", "10", "50"]:
            row[f"throughput_c{level}"] = data.get("throughput_tps", {}).get(level, 0)
            row[f"ttft_c{level}"] = data.get("ttft_ms", {}).get(level, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_pareto_data(results: dict) -> list[dict]:
    points = []
    for fmt_name, data in results.items():
        points.append({
            "format": fmt_name,
            "quality": data.get("mmlu_accuracy", 0),
            "speed": data.get("throughput_tps", {}).get("1", 0),
            "memory": data.get("vram_mb", 0),
        })
    return points


def plot_comparison(results: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt

    df = prepare_comparison_data(results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Quantization Format Comparison", fontsize=14)

    axes[0, 0].barh(df["format"], df["vram_mb"])
    axes[0, 0].set_xlabel("VRAM (MB)")
    axes[0, 0].set_title("Memory Usage")

    axes[0, 1].barh(df["format"], df["perplexity"])
    axes[0, 1].set_xlabel("Perplexity (lower = better)")
    axes[0, 1].set_title("Quality: Perplexity")

    axes[1, 0].barh(df["format"], df["throughput_c1"])
    axes[1, 0].set_xlabel("Tokens/sec (concurrency=1)")
    axes[1, 0].set_title("Throughput")

    axes[1, 1].barh(df["format"], df["ttft_c1"])
    axes[1, 1].set_xlabel("TTFT (ms)")
    axes[1, 1].set_title("Time to First Token")

    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=150)
    plt.close()


def plot_pareto(results: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt

    points = prepare_pareto_data(results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    for p in points:
        ax.scatter(p["speed"], p["quality"], s=p["memory"] / 20, alpha=0.7)
        ax.annotate(p["format"], (p["speed"], p["quality"]),
                    textcoords="offset points", xytext=(10, 5))

    ax.set_xlabel("Throughput (tokens/sec, concurrency=1)")
    ax.set_ylabel("MMLU Accuracy")
    ax.set_title("Quality vs Speed (bubble size = VRAM)")
    plt.tight_layout()
    plt.savefig(out / "pareto.png", dpi=150)
    plt.close()
