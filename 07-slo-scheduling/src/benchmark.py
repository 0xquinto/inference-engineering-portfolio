import asyncio
import json
import time

import httpx

from .metrics import RequestResult
from .workload import WorkloadRequest
from .scheduler import SLOScheduler


class SchedulingBenchmarker:
    def __init__(
        self, port: int = 8010, max_tokens: int = 512,
        temperature: float = 0.0, model_name: str = "default",
        policy: str = "fcfs",
        scheduler_config: dict | None = None,
    ):
        self.base_url = f"http://localhost:{port}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_name = model_name
        self.policy = policy
        self.scheduler = None
        if policy == "slo_aware" and scheduler_config:
            self.scheduler = SLOScheduler(
                max_queue_depth=scheduler_config.get("max_queue_depth", 50),
                max_concurrent=scheduler_config.get("max_concurrent", 10),
            )

    async def send_request(self, request: WorkloadRequest) -> RequestResult:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.policy == "priority":
            payload["extra_body"] = {"priority": request.priority}

        ttft = 0.0
        tokens = 0
        start = time.perf_counter()

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        if ttft == 0.0:
                            ttft = time.perf_counter() - start
                        tokens += 1

        latency = time.perf_counter() - start
        return RequestResult(
            request_class=request.request_class,
            slo_seconds=request.slo_seconds,
            latency_seconds=latency,
            ttft_seconds=ttft,
            tokens=tokens,
        )

    async def run_at_qps(
        self, requests: list[WorkloadRequest], qps: float,
    ) -> list[RequestResult]:
        interval = 1.0 / qps if qps > 0 else 0.0

        if self.policy == "slo_aware" and self.scheduler:
            return await self._run_scheduled(requests, interval)

        return await self._run_direct(requests, interval)

    async def _run_direct(
        self, requests: list[WorkloadRequest], interval: float,
    ) -> list[RequestResult]:
        tasks = []
        for i, req in enumerate(requests):
            if i > 0 and interval > 0:
                await asyncio.sleep(interval)
            tasks.append(asyncio.create_task(self.send_request(req)))
        return list(await asyncio.gather(*tasks))

    async def _run_scheduled(
        self, requests: list[WorkloadRequest], interval: float,
    ) -> list[RequestResult]:
        results: list[RequestResult] = []
        sem = asyncio.Semaphore(self.scheduler.max_concurrent)
        done_event = asyncio.Event()
        rejected_count = 0
        produced_count = 0

        async def producer():
            nonlocal rejected_count, produced_count
            for i, req in enumerate(requests):
                if i > 0 and interval > 0:
                    await asyncio.sleep(interval)
                if self.scheduler.enqueue(req):
                    rejected_count += 1
                produced_count += 1
            done_event.set()

        async def dispatch_one(req: WorkloadRequest):
            async with sem:
                result = await self.send_request(req)
                results.append(result)

        async def consumer():
            tasks = []
            while True:
                if self.scheduler.queue_size() > 0:
                    req = await self.scheduler.dequeue()
                    tasks.append(asyncio.create_task(dispatch_one(req)))
                elif done_event.is_set() and self.scheduler.queue_size() == 0:
                    break
                else:
                    await asyncio.sleep(0.01)
            await asyncio.gather(*tasks)

        await asyncio.gather(producer(), consumer())
        return results
