import pytest
import yaml
from pathlib import Path


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "configs" / "speculative.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_results():
    return {
        "baseline": {
            "1": {"ttft_p50": 42.0, "ttft_p95": 55.0, "throughput_p50": 85.0, "count": 50},
            "5": {"ttft_p50": 48.0, "ttft_p95": 62.0, "throughput_p50": 80.0, "count": 50},
            "10": {"ttft_p50": 55.0, "ttft_p95": 78.0, "throughput_p50": 72.0, "count": 50},
            "25": {"ttft_p50": 82.0, "ttft_p95": 120.0, "throughput_p50": 58.0, "count": 50},
            "50": {"ttft_p50": 145.0, "ttft_p95": 210.0, "throughput_p50": 42.0, "count": 50},
        },
        "eagle3": {
            "1": {"ttft_p50": 38.0, "ttft_p95": 50.0, "throughput_p50": 170.0, "count": 50},
            "5": {"ttft_p50": 44.0, "ttft_p95": 58.0, "throughput_p50": 155.0, "count": 50},
            "10": {"ttft_p50": 52.0, "ttft_p95": 72.0, "throughput_p50": 135.0, "count": 50},
            "25": {"ttft_p50": 78.0, "ttft_p95": 115.0, "throughput_p50": 105.0, "count": 50},
            "50": {"ttft_p50": 140.0, "ttft_p95": 200.0, "throughput_p50": 78.0, "count": 50},
        },
    }
