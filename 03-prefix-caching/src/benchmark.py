import asyncio
import json
import time

import httpx

from .metrics import RequestMetric, CacheMetrics


class CacheBenchmarker:
    def __init__(self, port: int = 8010, max_tokens: int = 128, temperature: float = 0.0, model_name: str = "default"):
        self.base_url = f"http://localhost:{port}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_name = model_name

    def _build_payload(self, messages: list[dict]) -> dict:
        return {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

    async def send_request(
        self, messages: list[dict], scenario: str, caching: bool, request_idx: int
    ) -> RequestMetric:
        payload = self._build_payload(messages)

        ttft_ms = 0.0
        tokens = 0
        start = time.perf_counter()

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload
            ) as resp:
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

        return RequestMetric(
            scenario=scenario,
            caching=caching,
            request_idx=request_idx,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens=tokens,
        )

    async def run_scenario_shared_system(
        self, requests: list[tuple[dict, dict]], scenario: str, caching: bool, concurrency: int
    ) -> list[RequestMetric]:
        sem = asyncio.Semaphore(concurrency)

        async def limited(idx: int, system: dict, user: dict) -> RequestMetric:
            async with sem:
                return await self.send_request([system, user], scenario, caching, idx)

        tasks = [limited(i, sys, usr) for i, (sys, usr) in enumerate(requests)]
        return list(await asyncio.gather(*tasks))

    async def run_scenario_multi_turn(
        self, conversations: list[list[list[dict]]], scenario: str, caching: bool
    ) -> list[RequestMetric]:
        results = []
        for conv_turns in conversations:
            for idx, messages in enumerate(conv_turns):
                metric = await self.send_request(messages, scenario, caching, idx)
                results.append(metric)
        return results

    async def run_scenario_cache_pressure(
        self, batches: list[list[tuple[dict, dict]]], scenario: str, caching: bool, concurrency: int
    ) -> dict[int, list[RequestMetric]]:
        results = {}
        for batch in batches:
            n_prefixes = len(set(sys["content"] for sys, _ in batch))
            metrics = await self.run_scenario_shared_system(batch, scenario, caching, concurrency)
            results[n_prefixes] = metrics
        return results
