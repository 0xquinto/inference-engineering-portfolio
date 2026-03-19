"""Tests for CLI entrypoint."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)


def test_help_flag():
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--help"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0
    assert "engine" in result.stdout
    assert "vllm" in result.stdout


def test_list_engines():
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--list-engines"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0
    assert "vllm" in result.stdout
    assert "sglang" in result.stdout
    assert "tensorrt-llm" in result.stdout
