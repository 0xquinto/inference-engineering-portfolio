"""SGLang benchmark runner."""

import os
import subprocess

from .base import BenchmarkRunner

SGLANG_VENV_PYTHON = "/workspace/venvs/sglang/bin/python"


class SglangRunner(BenchmarkRunner):
    """Runs SGLang as a subprocess and benchmarks it."""

    def __init__(self, port: int = 8002):
        super().__init__("sglang", port)

    async def start_server(self, model: str, extra_args: list[str] | None = None) -> None:
        python_bin = SGLANG_VENV_PYTHON if os.path.exists(SGLANG_VENV_PYTHON) else "python"
        cmd = [
            python_bin, "-m", "sglang.launch_server",
            "--model-path", model,
            "--port", str(self.port),
            *(extra_args or []),
        ]
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        await self.wait_for_server()

    async def stop_server(self) -> None:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
