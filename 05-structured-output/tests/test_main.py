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
        assert "Structured Output" in result.stdout

    def test_list_backends(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--list-backends"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "xgrammar" in result.stdout
        assert "unconstrained" in result.stdout

    def test_list_schemas(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--list-schemas"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "simple_json" in result.stdout
        assert "function_call" in result.stdout
