from pathlib import Path
import pytest
from src.profiles import load_profile

def test_load_gpu_profile():
    profile = load_profile("gpu", Path(__file__).parent.parent / "profiles")
    assert "engines" in profile

def test_load_local_profile():
    profile = load_profile("local", Path(__file__).parent.parent / "profiles")
    assert profile["model"]["name"] == "Qwen/Qwen3.5-4B"
    assert "vllm" in profile["engines"]
    assert profile["engines"]["vllm"]["port"] == 11434
    assert profile["engines"]["vllm"]["model_id"] == "qwen3.5:4b"

def test_invalid_profile_raises():
    with pytest.raises(FileNotFoundError):
        load_profile("nonexistent", Path(__file__).parent.parent / "profiles")
