import pytest
from pathlib import Path

import yaml


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "configs" / "scheduling.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_results():
    return {
        "fcfs": {
            "1": {"goodput": 0.95, "ttft_p50": 80, "ttft_p95": 200,
                   "latency_p50": 1.2, "latency_p95": 3.5, "throughput": 10.0,
                   "fairness": 0.9},
            "10": {"goodput": 0.60, "ttft_p50": 400, "ttft_p95": 2000,
                    "latency_p50": 5.0, "latency_p95": 15.0, "throughput": 8.0,
                    "fairness": 0.4},
        },
        "slo_aware": {
            "1": {"goodput": 0.95, "ttft_p50": 80, "ttft_p95": 200,
                   "latency_p50": 1.2, "latency_p95": 3.5, "throughput": 10.0,
                   "fairness": 0.9},
            "10": {"goodput": 0.90, "ttft_p50": 150, "ttft_p95": 500,
                    "latency_p50": 3.0, "latency_p95": 8.0, "throughput": 9.0,
                    "fairness": 0.85},
        },
    }
