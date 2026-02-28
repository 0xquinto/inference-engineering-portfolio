"""vLLM benchmark runner."""

import os
import shutil
import subprocess

from .base import BenchmarkRunner

VLLM_VENV_BIN = "/workspace/venvs/vllm/bin/vllm"


class VllmRunner(BenchmarkRunner):
    """Runs vLLM as a subprocess and benchmarks it."""

    def __init__(self, port: int = 8001):
        super().__init__("vllm", port)

    async def start_server(self, model: str, extra_args: list[str] | None = None) -> None:
        vllm_bin = VLLM_VENV_BIN if os.path.exists(VLLM_VENV_BIN) else shutil.which("vllm") or "vllm"
        cmd = [
            vllm_bin, "serve", model,
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
