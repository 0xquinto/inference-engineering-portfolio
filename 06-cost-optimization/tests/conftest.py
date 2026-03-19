import pytest
import yaml
from pathlib import Path


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "configs" / "cost.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_cost_table():
    return [
        {
            "model": "small",
            "tps": 250.0,
            "gpu_cost_per_hour": 0.75,
            "cost_per_million": 0.83,
            "monthly_cost": 540.0,
            "monthly_tokens": 648_000_000,
        },
        {
            "model": "medium",
            "tps": 85.0,
            "gpu_cost_per_hour": 0.75,
            "cost_per_million": 2.45,
            "monthly_cost": 540.0,
            "monthly_tokens": 220_320_000,
        },
        {
            "model": "large",
            "tps": 25.0,
            "gpu_cost_per_hour": 3.59,
            "cost_per_million": 39.89,
            "monthly_cost": 2584.80,
            "monthly_tokens": 64_800_000,
        },
    ]
