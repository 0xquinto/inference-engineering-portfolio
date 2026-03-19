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
