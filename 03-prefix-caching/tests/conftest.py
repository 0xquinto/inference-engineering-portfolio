import pytest
import yaml
from pathlib import Path


@pytest.fixture
def scenario_config():
    path = Path(__file__).parent.parent / "configs" / "scenarios.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def engines_config():
    path = Path(__file__).parent.parent / "configs" / "engines.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
