import subprocess
import sys


class TestCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True, text=True, cwd=".",
        )
        assert result.returncode == 0
        assert "Quantization Pipeline" in result.stdout

    def test_list_formats(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--list-formats"],
            capture_output=True, text=True, cwd=".",
        )
        assert result.returncode == 0
        assert "bf16" in result.stdout
        assert "w4a16" in result.stdout
