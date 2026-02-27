"""
Abstract benchmark runner.

Start here — implement this interface for each engine (vLLM, SGLang, TGI).
Each runner handles: starting the server, sending requests, collecting metrics, and teardown.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    engine: str
    prompt_category: str  # short, medium, long, code
    concurrency: int
    ttft_ms: float        # time to first token
    total_time_ms: float
    tokens_generated: int
    tokens_per_sec: float
    gpu_memory_mb: float


class BenchmarkRunner(ABC):
    """Base class for all inference engine benchmark runners."""

    @abstractmethod
    async def start_server(self, model: str, **kwargs) -> None:
        """Start the inference server with given model and config."""
        ...

    @abstractmethod
    async def stop_server(self) -> None:
        """Gracefully stop the inference server."""
        ...

    @abstractmethod
    async def send_request(self, prompt: str, max_tokens: int = 256) -> BenchmarkResult:
        """Send a single inference request and measure metrics."""
        ...

    @abstractmethod
    async def get_gpu_memory_usage(self) -> float:
        """Return current GPU memory usage in MB."""
        ...

    async def run_benchmark(
        self, prompts: list[dict], concurrency_levels: list[int]
    ) -> list[BenchmarkResult]:
        """Run full benchmark suite across concurrency levels."""
        # TODO: Implement concurrent request sending with asyncio.gather
        # TODO: Warm up the model with a few requests first
        # TODO: Collect results per prompt category and concurrency level
        results = []
        return results
