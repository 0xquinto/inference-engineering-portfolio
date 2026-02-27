"""Tests for visualization (no display needed — tests chart data generation)."""

from src.visualization.plots import prepare_latency_data, prepare_throughput_data


def test_prepare_latency_data():
    results = {
        "vllm": [
            {"engine": "vllm", "category": "short", "concurrency": 1,
             "latency": {"p50": 100, "p95": 200, "p99": 300, "mean": 150, "count": 5}},
        ],
        "sglang": [
            {"engine": "sglang", "category": "short", "concurrency": 1,
             "latency": {"p50": 80, "p95": 160, "p99": 240, "mean": 120, "count": 5}},
        ],
    }
    df = prepare_latency_data(results)
    assert len(df) == 2
    assert "engine" in df.columns
    assert "p50" in df.columns


def test_prepare_throughput_data():
    results = {
        "vllm": [
            {"engine": "vllm", "category": "short", "concurrency": 1,
             "throughput": {"avg_tokens_per_sec": 500, "total_requests": 5}},
            {"engine": "vllm", "category": "short", "concurrency": 10,
             "throughput": {"avg_tokens_per_sec": 4000, "total_requests": 50}},
        ],
    }
    df = prepare_throughput_data(results)
    assert len(df) == 2
    assert "concurrency" in df.columns
