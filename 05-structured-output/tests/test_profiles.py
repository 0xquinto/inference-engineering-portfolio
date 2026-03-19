from pathlib import Path
import pytest
from src.profiles import load_profile

class TestLoadProfile:
    def test_load_gpu_profile(self):
        profile = load_profile("gpu", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-9B"
        assert "xgrammar" in profile["backends"]

    def test_load_local_profile(self):
        profile = load_profile("local", Path(__file__).parent.parent / "profiles")
        assert profile["model"]["name"] == "Qwen/Qwen3.5-4B"
        assert max(profile["benchmark"]["concurrency_levels"]) <= 10

    def test_invalid_profile_raises(self):
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent", Path(__file__).parent.parent / "profiles")
