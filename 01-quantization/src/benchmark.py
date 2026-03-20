import asyncio
import json
import time
from dataclasses import dataclass

import httpx


@dataclass
class BenchmarkResult:
    format_name: str
    concurrency: int
    ttft_ms: float
    total_time_ms: float
    tokens_generated: int
    gpu_memory_mb: int

    @property
    def tokens_per_sec(self) -> float:
        if self.total_time_ms <= 0:
            return 0.0
        return self.tokens_generated / (self.total_time_ms / 1000)

    def to_dict(self) -> dict:
        return {
            "format_name": self.format_name,
            "concurrency": self.concurrency,
            "ttft_ms": self.ttft_ms,
            "total_time_ms": self.total_time_ms,
            "tokens_generated": self.tokens_generated,
            "tokens_per_sec": self.tokens_per_sec,
            "gpu_memory_mb": self.gpu_memory_mb,
        }

    @staticmethod
    def aggregate(results: list["BenchmarkResult"]) -> dict:
        if not results:
            return {"count": 0}
        ttfts = sorted(r.ttft_ms for r in results)
        tps_list = sorted(r.tokens_per_sec for r in results)
        n = len(results)
        return {
            "count": n,
            "ttft_p50": ttfts[n // 2],
            "ttft_p95": ttfts[int(n * 0.95)] if n >= 20 else ttfts[-1],
            "throughput_p50": tps_list[n // 2],
            "avg_gpu_memory_mb": sum(r.gpu_memory_mb for r in results) // n,
        }


class PerfBenchmarker:
    def __init__(self, port: int = 8010, max_tokens: int = 256, temperature: float = 0.0, model_name: str = "default"):
        self.base_url = f"http://localhost:{port}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_name = model_name

    async def send_request(self, prompt: str, format_name: str, concurrency: int) -> BenchmarkResult:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        ttft_ms = 0.0
        tokens = 0
        start = time.perf_counter()

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content") or delta.get("reasoning"):
                        if ttft_ms == 0.0:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        tokens += 1

        total_ms = (time.perf_counter() - start) * 1000
        gpu_mem = self._get_gpu_memory()

        return BenchmarkResult(
            format_name=format_name,
            concurrency=concurrency,
            ttft_ms=ttft_ms,
            total_time_ms=total_ms,
            tokens_generated=tokens,
            gpu_memory_mb=gpu_mem,
        )

    async def run_concurrent(
        self, prompts: list[str], concurrency: int, format_name: str
    ) -> list[BenchmarkResult]:
        sem = asyncio.Semaphore(concurrency)

        async def limited(prompt: str) -> BenchmarkResult:
            async with sem:
                return await self.send_request(prompt, format_name, concurrency)

        tasks = [limited(p) for p in prompts]
        return list(await asyncio.gather(*tasks))

    @staticmethod
    def _get_gpu_memory() -> int:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return int(result.stdout.strip().split("\n")[0])
        except Exception:
            return 0

    @staticmethod
    def aggregate(results: list[BenchmarkResult]) -> dict:
        return BenchmarkResult.aggregate(results)
