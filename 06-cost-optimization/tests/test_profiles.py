from pathlib import Path
import pytest
from src.profiles import load_profile

class TestLoadProfile:
    def test_load_gpu_profile(self):
        profile = load_profile("gpu", Path(__file__).parent.parent / "profiles")
        assert profile["models"]["small"]["gpu_cost_per_hour"] == 0.75

    def test_load_local_profile(self):
        profile = load_profile("local", Path(__file__).parent.parent / "profiles")
        assert profile["models"]["local_small"]["gpu_cost_per_hour"] == 0.0
        assert profile["models"]["local_medium"]["gpu_cost_per_hour"] == 0.0
        assert profile["models"]["cloud"]["gpu_cost_per_hour"] > 0
        assert profile["models"]["local_small"]["port"] == 11434
        assert profile["models"]["local_small"]["model_id"] == "qwen3.5:0.8b"
        assert profile["models"]["local_medium"]["model_id"] == "qwen3.5:4b"
        assert profile["models"]["cloud"]["model_id"] == "default"

    def test_invalid_profile_raises(self):
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent", Path(__file__).parent.parent / "profiles")
