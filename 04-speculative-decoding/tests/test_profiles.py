from pathlib import Path
import pytest
from src.profiles import load_profile


class TestLoadProfile:
    def test_load_gpu_profile(self):
        profile = load_profile("gpu", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-9B"
        assert profile["benchmark"]["qps_levels"] == [1, 5, 10, 25, 50]

    def test_load_local_profile(self):
        profile = load_profile("local", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-4B"
        assert max(profile["benchmark"]["qps_levels"]) <= 10
        assert profile["benchmark"]["port"] == 11434
        assert profile["benchmark"]["model_id"] == "qwen3.5:4b"

    def test_invalid_profile_raises(self):
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent", Path(__file__).parent.parent / "profiles")
