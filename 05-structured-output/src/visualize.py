from pathlib import Path

import pandas as pd


def prepare_tps_data(results: dict) -> pd.DataFrame:
    rows = []
    for backend_name, schema_data in results.items():
        for schema_name, metrics in schema_data.items():
            rows.append({
                "backend": backend_name,
                "schema": schema_name,
                "throughput_p50": metrics.get("throughput_p50", 0),
            })
    return pd.DataFrame(rows)


def prepare_validity_data(results: dict) -> pd.DataFrame:
    rows = []
    for backend_name, schema_data in results.items():
        for schema_name, metrics in schema_data.items():
            rows.append({
                "backend": backend_name,
                "schema": schema_name,
                "validity_rate": metrics.get("validity_rate", 0) * 100,
            })
    return pd.DataFrame(rows)


def prepare_latency_data(results: dict, schema_name: str) -> pd.DataFrame:
    rows = []
    for backend_name, schema_data in results.items():
        if schema_name in schema_data:
            metrics = schema_data[schema_name]
            rows.append({
                "backend": backend_name,
                "ttft_p50": metrics.get("ttft_p50", 0),
            })
    return pd.DataFrame(rows)


def plot_tps_overhead(results: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt

    df = prepare_tps_data(results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    schemas = df["schema"].unique()
    backends = df["backend"].unique()
    n_schemas = len(schemas)
    n_backends = len(backends)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.8 / n_backends
    x = range(n_schemas)

    for i, backend in enumerate(backends):
        subset = df[df["backend"] == backend]
        values = [
            subset[subset["schema"] == s]["throughput_p50"].values[0]
            if len(subset[subset["schema"] == s]) > 0 else 0
            for s in schemas
        ]
        offset = (i - n_backends / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], values, width, label=backend)

    ax.set_xticks(list(x))
    ax.set_xticklabels(schemas, rotation=15, ha="right")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput by Schema Complexity and Backend")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out / "tps_overhead.png", dpi=150)
    plt.close()


def plot_validity_rate(results: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt

    df = prepare_validity_data(results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    schemas = df["schema"].unique()
    backends = df["backend"].unique()
    n_schemas = len(schemas)
    n_backends = len(backends)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.8 / n_backends
    x = range(n_schemas)

    for i, backend in enumerate(backends):
        subset = df[df["backend"] == backend]
        values = [
            subset[subset["schema"] == s]["validity_rate"].values[0]
            if len(subset[subset["schema"] == s]) > 0 else 0
            for s in schemas
        ]
        offset = (i - n_backends / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], values, width, label=backend)

    ax.set_xticks(list(x))
    ax.set_xticklabels(schemas, rotation=15, ha="right")
    ax.set_ylabel("Validity Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Output Validity by Schema Complexity and Backend")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out / "validity_rate.png", dpi=150)
    plt.close()


def plot_latency_comparison(results: dict, output_dir: str = "results/") -> None:
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    schemas = list(next(iter(results.values())).keys()) if results else []
    if not schemas:
        return

    schema_name = schemas[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    for backend_name, schema_data in results.items():
        if schema_name in schema_data:
            metrics = schema_data[schema_name]
            ttft = metrics.get("ttft_p50", 0)
            ax.barh(backend_name, ttft)

    ax.set_xlabel("TTFT p50 (ms)")
    ax.set_title(f"Latency Comparison — {schema_name}")

    plt.tight_layout()
    plt.savefig(out / "latency_comparison.png", dpi=150)
    plt.close()
