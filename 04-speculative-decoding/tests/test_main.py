import json
import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

import matplotlib
matplotlib.use("Agg")

from src.config import load_config
from src.main import async_main

PROJECT_ROOT = str(Path(__file__).parent.parent)
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "speculative.yaml"


def _make_args(tmp_path, method=None, step="benchmark"):
    return type("Args", (), {
        "profile": None,
        "config": str(CONFIG_PATH),
        "output": str(tmp_path),
        "method": method,
        "step": step,
        "list_methods": False,
    })()


class TestCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "Speculative Decoding" in result.stdout

    def test_list_methods(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--list-methods"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "baseline" in result.stdout
        assert "eagle3" in result.stdout
        assert "ngram" in result.stdout


class TestResultAccumulation:
    def test_single_method_preserves_existing_results(self, tmp_path):
        """Running --method ngram should keep existing baseline results."""
        results_file = tmp_path / "speculative_results.json"
        existing = {
            "baseline": {
                "1": {"ttft_p50": 180.0, "throughput_p50": 28.0, "count": 50}
            }
        }
        results_file.write_text(json.dumps(existing))

        args = _make_args(tmp_path, method="ngram", step="benchmark")
        cfg = load_config(CONFIG_PATH)

        with patch("src.benchmark.SpecBenchmarker.run_at_qps", new_callable=AsyncMock, return_value=[]):
            asyncio.run(async_main(args, cfg))

        saved = json.loads(results_file.read_text())
        assert "baseline" in saved, "Existing baseline results should be preserved"
        assert "ngram" in saved, "New ngram results should be added"

    def test_visualize_loads_from_file(self, tmp_path):
        """Running --step visualize should load results from file and generate charts."""
        results_file = tmp_path / "speculative_results.json"
        existing = {
            "baseline": {
                "1": {"ttft_p50": 180.0, "throughput_p50": 28.0, "count": 50},
                "5": {"ttft_p50": 177.0, "throughput_p50": 28.6, "count": 50},
            },
            "ngram": {
                "1": {"ttft_p50": 160.0, "throughput_p50": 32.0, "count": 50},
                "5": {"ttft_p50": 155.0, "throughput_p50": 31.0, "count": 50},
            },
        }
        results_file.write_text(json.dumps(existing))

        args = _make_args(tmp_path, method=None, step="visualize")
        cfg = load_config(CONFIG_PATH)
        asyncio.run(async_main(args, cfg))

        saved = json.loads(results_file.read_text())
        assert "baseline" in saved
        assert "ngram" in saved
        assert saved["baseline"]["1"]["throughput_p50"] == 28.0

        assert (tmp_path / "ttft_by_qps.png").exists()
        assert (tmp_path / "throughput_by_qps.png").exists()

    def test_visualize_does_not_overwrite_results(self, tmp_path):
        """Visualize step should preserve existing results, not overwrite with empty."""
        results_file = tmp_path / "speculative_results.json"
        existing = {"baseline": {"1": {"ttft_p50": 100.0, "throughput_p50": 30.0, "count": 50}}}
        results_file.write_text(json.dumps(existing))

        args = _make_args(tmp_path, method=None, step="visualize")
        cfg = load_config(CONFIG_PATH)
        asyncio.run(async_main(args, cfg))

        saved = json.loads(results_file.read_text())
        assert saved["baseline"]["1"]["throughput_p50"] == 30.0, "Results should not be overwritten"
