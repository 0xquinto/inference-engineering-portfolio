import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)


class TestCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "Cost Optimization" in result.stdout

    def test_list_models(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--list-models"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "small" in result.stdout
        assert "medium" in result.stdout
        assert "large" in result.stdout
