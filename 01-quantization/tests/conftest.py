import pytest
import yaml
from pathlib import Path


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "configs" / "quantization.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_results():
    return {
        "bf16": {
            "perplexity": 6.82,
            "mmlu_accuracy": 0.714,
            "ttft_ms": {"1": 45.2, "10": 52.1, "50": 89.3},
            "throughput_tps": {"1": 83.1, "10": 412.5, "50": 1021.0},
            "vram_mb": 14200,
            "load_time_s": 12.3,
        },
        "gptq_int4": {
            "perplexity": 7.01,
            "mmlu_accuracy": 0.698,
            "ttft_ms": {"1": 32.1, "10": 38.5, "50": 62.7},
            "throughput_tps": {"1": 112.4, "10": 567.8, "50": 1456.0},
            "vram_mb": 5100,
            "load_time_s": 8.1,
        },
    }
