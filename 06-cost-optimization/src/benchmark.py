import asyncio
import json
import time
from dataclasses import dataclass

import httpx

from .cascade import CascadeRouter, CascadeDecision


@dataclass
class CascadeResult:
    model_name: str
    complexity: str
    prompt: str
    ttft_ms: float
    total_time_ms: float
    tokens_generated: int
    quality_score: float
    escalated: bool

    @property
    def tokens_per_sec(self) -> float:
        if self.total_time_ms <= 0:
            return 0.0
        return self.tokens_generated / (self.total_time_ms / 1000)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "complexity": self.complexity,
            "prompt": self.prompt,
            "ttft_ms": self.ttft_ms,
            "total_time_ms": self.total_time_ms,
            "tokens_generated": self.tokens_generated,
            "tokens_per_sec": self.tokens_per_sec,
            "quality_score": self.quality_score,
            "escalated": self.escalated,
        }


class CascadeBenchmarker:
    def __init__(
        self,
        models: dict[str, int],
        max_tokens: int = 256,
        temperature: float = 0.0,
        model_ids: dict[str, str] | None = None,
    ):
        """Initialize benchmarker.

        Args:
            models: Maps tier name (e.g. "small") to port number.
            max_tokens: Max tokens to generate per request.
            temperature: Sampling temperature.
            model_ids: Maps tier name to API model field. Defaults to "default" for all.
        """
        self.models = models
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_ids = model_ids or {}

    async def send_request(
        self, prompt: str, port: int, model_name: str, complexity: str
    ) -> CascadeResult:
        payload = {
            "model": self.model_ids.get(model_name, "default"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        ttft_ms = 0.0
        tokens = 0
        start = time.perf_counter()

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"http://localhost:{port}/v1/chat/completions", json=payload
            ) as resp:
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

        return CascadeResult(
            model_name=model_name,
            complexity=complexity,
            prompt=prompt,
            ttft_ms=ttft_ms,
            total_time_ms=total_ms,
            tokens_generated=tokens,
            quality_score=0.0,
            escalated=False,
        )

    async def run_cascade(
        self,
        prompts_by_complexity: dict[str, list[str]],
        router: CascadeRouter,
        concurrency: int = 10,
    ) -> list[CascadeResult]:
        sem = asyncio.Semaphore(concurrency)
        tasks = []

        for complexity, prompts in prompts_by_complexity.items():
            for prompt in prompts:
                decision = router.route(prompt)
                port = self.models.get(decision.routed_to, 8010)

                async def limited(p=prompt, pt=port, mn=decision.routed_to, cx=complexity):
                    async with sem:
                        return await self.send_request(p, pt, mn, cx)

                tasks.append(limited())

        return list(await asyncio.gather(*tasks))

    @staticmethod
    def aggregate(results: list[CascadeResult]) -> dict:
        if not results:
            return {"count": 0}
        ttfts = sorted(r.ttft_ms for r in results)
        tps_list = sorted(r.tokens_per_sec for r in results)
        n = len(results)
        by_model: dict[str, list[CascadeResult]] = {}
        for r in results:
            by_model.setdefault(r.model_name, []).append(r)

        per_model = {}
        for model_name, model_results in by_model.items():
            m_ttfts = sorted(r.ttft_ms for r in model_results)
            m_tps = sorted(r.tokens_per_sec for r in model_results)
            mn = len(model_results)
            per_model[model_name] = {
                "count": mn,
                "ttft_p50": m_ttfts[mn // 2],
                "throughput_p50": m_tps[mn // 2],
            }

        return {
            "count": n,
            "ttft_p50": ttfts[n // 2],
            "ttft_p95": ttfts[int(n * 0.95)] if n >= 20 else ttfts[-1],
            "throughput_p50": tps_list[n // 2],
            "per_model": per_model,
        }
