import subprocess
import sys


class TestCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True, text=True, cwd=".",
        )
        assert result.returncode == 0
        assert "Prefix Caching" in result.stdout

    def test_list_scenarios(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--list-scenarios"],
            capture_output=True, text=True, cwd=".",
        )
        assert result.returncode == 0
        assert "shared_system_prompt" in result.stdout
        assert "cache_pressure" in result.stdout
