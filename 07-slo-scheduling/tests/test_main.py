import subprocess
import sys


class TestCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True, text=True, cwd=".",
        )
        assert result.returncode == 0
        assert "SLO-Aware" in result.stdout

    def test_list_policies(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--list-policies"],
            capture_output=True, text=True, cwd=".",
        )
        assert result.returncode == 0
        assert "fcfs" in result.stdout
        assert "slo_aware" in result.stdout
