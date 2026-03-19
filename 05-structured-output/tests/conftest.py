import pytest
import yaml
from pathlib import Path


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "configs" / "structured.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_results():
    return {
        "xgrammar": {
            "simple_json": {
                "ttft_p50": 35.0, "throughput_p50": 95.0,
                "validity_rate": 1.0, "avg_retries": 0.0,
            },
            "nested_object": {
                "ttft_p50": 42.0, "throughput_p50": 78.0,
                "validity_rate": 0.98, "avg_retries": 0.0,
            },
            "function_call": {
                "ttft_p50": 48.0, "throughput_p50": 70.0,
                "validity_rate": 0.95, "avg_retries": 0.0,
            },
        },
        "unconstrained": {
            "simple_json": {
                "ttft_p50": 30.0, "throughput_p50": 105.0,
                "validity_rate": 0.85, "avg_retries": 0.4,
            },
            "nested_object": {
                "ttft_p50": 32.0, "throughput_p50": 98.0,
                "validity_rate": 0.60, "avg_retries": 1.2,
            },
            "function_call": {
                "ttft_p50": 33.0, "throughput_p50": 95.0,
                "validity_rate": 0.45, "avg_retries": 2.1,
            },
        },
    }
