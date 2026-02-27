"""
Abstract benchmark runner interface.

Each engine (vLLM, SGLang, TensorRT-LLM) implements this interface.
The runner handles: starting the server, sending requests, collecting metrics, and teardown.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx


@dataclass
class RequestConfig:
    max_tokens: int = 256
    temperature: float = 0.0


@dataclass
class BenchmarkResult:
    engine: str
    prompt_category: str
    concurrency: int
    ttft_ms: float
    total_time_ms: float
    tokens_generated: int
    tokens_per_sec: float
    gpu_memory_mb: float


class BenchmarkRunner(ABC):
    """Base class for all inference engine benchmark runners."""

    def __init__(self, engine_name: str, port: int):
        self.engine_name = engine_name
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self._process = None

    @abstractmethod
    async def start_server(self, model: str, extra_args: list[str] | None = None) -> None:
        """Start the inference server."""
        ...

    @abstractmethod
    async def stop_server(self) -> None:
        """Stop the inference server."""
        ...

    async def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the server is ready."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/health", timeout=timeout)
                return resp.status_code == 200
        except httpx.RequestError:
            return False

    async def wait_for_server(self, max_wait: int = 300, interval: int = 5):
        """Wait until the server is ready, up to max_wait seconds."""
        for _ in range(max_wait // interval):
            if await self.health_check():
                return True
            await asyncio.sleep(interval)
        raise TimeoutError(f"{self.engine_name} server did not start within {max_wait}s")

    async def send_request(
        self,
        prompt: str,
        config: RequestConfig | None = None,
    ) -> BenchmarkResult:
        """Send a single chat completion request and measure metrics."""
        cfg = config or RequestConfig()
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "stream": True,
        }

        start = time.perf_counter()
        ttft = None
        tokens = 0

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line.strip() == "data: [DONE]":
                        break
                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000
                    tokens += 1

        total_ms = (time.perf_counter() - start) * 1000
        ttft = ttft or total_ms
        tps = (tokens / (total_ms / 1000)) if total_ms > 0 else 0

        return BenchmarkResult(
            engine=self.engine_name,
            prompt_category="",
            concurrency=1,
            ttft_ms=round(ttft, 2),
            total_time_ms=round(total_ms, 2),
            tokens_generated=tokens,
            tokens_per_sec=round(tps, 2),
            gpu_memory_mb=0,
        )

    async def run_concurrent(
        self,
        prompts: list[str],
        concurrency: int,
        config: RequestConfig | None = None,
    ) -> list[BenchmarkResult]:
        """Send requests at the given concurrency level."""
        sem = asyncio.Semaphore(concurrency)

        async def limited_request(prompt: str) -> BenchmarkResult:
            async with sem:
                return await self.send_request(prompt, config)

        tasks = [limited_request(p) for p in prompts]
        results = await asyncio.gather(*tasks)
        for r in results:
            r.concurrency = concurrency
        return list(results)
