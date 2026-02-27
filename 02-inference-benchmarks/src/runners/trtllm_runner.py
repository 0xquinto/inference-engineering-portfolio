"""TensorRT-LLM benchmark runner (via Triton Inference Server Docker container)."""

import subprocess

from .base import BenchmarkRunner


class TrtllmRunner(BenchmarkRunner):
    """Runs TensorRT-LLM via Docker and benchmarks it."""

    def __init__(self, port: int = 8003, docker_image: str = "nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3"):
        super().__init__("tensorrt-llm", port)
        self.docker_image = docker_image
        self._container_id = None

    async def start_server(self, model: str, extra_args: list[str] | None = None) -> None:
        cmd = [
            "docker", "run", "-d", "--gpus", "all",
            "-p", f"{self.port}:8000",
            "--name", "trtllm-bench",
            self.docker_image,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self._container_id = result.stdout.strip()
        await self.wait_for_server(max_wait=600)

    async def stop_server(self) -> None:
        if self._container_id:
            subprocess.run(["docker", "stop", "trtllm-bench"], capture_output=True)
            subprocess.run(["docker", "rm", "trtllm-bench"], capture_output=True)
            self._container_id = None
