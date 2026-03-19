import asyncio
import json
import time
from dataclasses import dataclass

import httpx

from .methods import MethodResult


class SpecBenchmarker:
    def __init__(self, port: int = 8010, max_tokens: int = 256, temperature: float = 0.0, model_name: str = "default"):
        self.base_url = f"http://localhost:{port}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_name = model_name

    async def send_request(self, prompt: str, method_name: str, qps: int) -> MethodResult:
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
                    if delta.get("content"):
                        if ttft_ms == 0.0:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        tokens += 1

        total_ms = (time.perf_counter() - start) * 1000

        return MethodResult(
            method_name=method_name,
            qps=qps,
            ttft_ms=ttft_ms,
            total_time_ms=total_ms,
            tokens_generated=tokens,
            acceptance_rate=None,
        )

    async def run_at_qps(
        self, prompts: list[str], qps: int, method_name: str
    ) -> list[MethodResult]:
        tasks = []
        delay = 1.0 / qps if qps > 0 else 0.0

        for i, prompt in enumerate(prompts):
            if i > 0 and delay > 0:
                await asyncio.sleep(delay)
            task = asyncio.create_task(self.send_request(prompt, method_name, qps))
            tasks.append(task)

        return list(await asyncio.gather(*tasks))

    @staticmethod
    def aggregate(results: list[MethodResult]) -> dict:
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
        }
