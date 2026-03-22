import asyncio
import json
import re
import time
from dataclasses import dataclass

import httpx

from .schemas import validate_output


@dataclass
class StructuredResult:
    backend_name: str
    schema_name: str
    concurrency: int
    ttft_ms: float
    total_time_ms: float
    tokens_generated: int
    valid: bool
    retries: int

    @property
    def tokens_per_sec(self) -> float:
        if self.total_time_ms <= 0:
            return 0.0
        return self.tokens_generated / (self.total_time_ms / 1000)

    def to_dict(self) -> dict:
        return {
            "backend_name": self.backend_name,
            "schema_name": self.schema_name,
            "concurrency": self.concurrency,
            "ttft_ms": self.ttft_ms,
            "total_time_ms": self.total_time_ms,
            "tokens_generated": self.tokens_generated,
            "tokens_per_sec": self.tokens_per_sec,
            "valid": self.valid,
            "retries": self.retries,
        }


class StructuredBenchmarker:
    def __init__(
        self, port: int = 8010, max_tokens: int = 512,
        temperature: float = 0.0, max_retries: int = 3,
        model_name: str = "default",
        schema_format: str = "guided_json",
        disable_thinking: bool = False,
    ):
        self.base_url = f"http://localhost:{port}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.model_name = model_name
        self.schema_format = schema_format
        self.disable_thinking = disable_thinking

    async def send_request(
        self, prompt: str, schema: dict | None, backend_name: str,
        schema_name: str, concurrency: int,
    ) -> StructuredResult:
        is_constrained = schema is not None and backend_name != "unconstrained"
        retries = 0
        valid = False
        total_ttft_ms = 0.0
        total_time_ms = 0.0
        total_tokens = 0

        max_attempts = 1 if is_constrained else self.max_retries + 1

        for attempt in range(max_attempts):
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True,
            }
            if self.disable_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            if is_constrained and schema is not None:
                if self.schema_format == "response_format":
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        },
                    }
                else:
                    payload["guided_json"] = schema

            ttft_ms = 0.0
            tokens = 0
            content_parts = []
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
                        token_text = delta.get("content") or delta.get("reasoning")
                        if token_text:
                            if ttft_ms == 0.0:
                                ttft_ms = (time.perf_counter() - start) * 1000
                            tokens += 1
                            if delta.get("content"):
                                content_parts.append(delta["content"])
                            elif self.disable_thinking and delta.get("reasoning"):
                                # Workaround: vLLM streaming parser may route
                                # content to reasoning field when thinking is
                                # disabled (vLLM PR #37414)
                                content_parts.append(delta["reasoning"])

            elapsed_ms = (time.perf_counter() - start) * 1000

            if attempt == 0:
                total_ttft_ms = ttft_ms
            total_time_ms += elapsed_ms
            total_tokens += tokens

            output_text = "".join(content_parts)
            # Strip <think>...</think> tags from thinking models (e.g. Qwen3.5)
            output_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()
            valid = validate_output(schema_name, output_text)

            if valid:
                break
            if not is_constrained:
                retries += 1

        return StructuredResult(
            backend_name=backend_name,
            schema_name=schema_name,
            concurrency=concurrency,
            ttft_ms=total_ttft_ms,
            total_time_ms=total_time_ms,
            tokens_generated=total_tokens,
            valid=valid,
            retries=retries,
        )

    async def run_concurrent(
        self, prompts: list[str], concurrency: int, schema: dict | None,
        backend_name: str, schema_name: str,
    ) -> list[StructuredResult]:
        sem = asyncio.Semaphore(concurrency)

        async def limited(prompt: str) -> StructuredResult:
            async with sem:
                return await self.send_request(
                    prompt, schema, backend_name, schema_name, concurrency,
                )

        tasks = [limited(p) for p in prompts]
        return list(await asyncio.gather(*tasks))

    @staticmethod
    def aggregate(results: list[StructuredResult]) -> dict:
        if not results:
            return {"count": 0}
        ttfts = sorted(r.ttft_ms for r in results)
        tps_list = sorted(r.tokens_per_sec for r in results)
        n = len(results)
        valid_count = sum(1 for r in results if r.valid)
        total_retries = sum(r.retries for r in results)
        return {
            "count": n,
            "ttft_p50": ttfts[n // 2],
            "ttft_p95": ttfts[int(n * 0.95)] if n >= 20 else ttfts[-1],
            "throughput_p50": tps_list[n // 2],
            "validity_rate": valid_count / n,
            "avg_retries": total_retries / n,
        }
