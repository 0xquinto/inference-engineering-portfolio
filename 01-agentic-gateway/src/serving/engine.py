"""
vLLM engine wrapper.

Manages multiple vLLM AsyncLLMEngine instances (one per model)
and exposes a unified interface for the gateway to call.
"""

import asyncio
import time
from dataclasses import dataclass

import httpx

from .models import ModelConfig


@dataclass
class GenerationResult:
    model_key: str
    text: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    total_time_ms: float
    finish_reason: str


class InferenceEngine:
    """
    Wraps vLLM servers running as separate processes.

    Architecture: each model runs as a standalone vLLM server on a different port.
    This class handles routing requests to the correct server and collecting metrics.

    Why separate processes instead of AsyncLLMEngine in-process?
    - Simpler to manage GPU memory (each server controls its own allocation)
    - Can restart one model without affecting the other
    - Mirrors how you'd actually deploy in production
    """

    def __init__(self):
        self._servers: dict[str, str] = {}  # model_key -> base_url
        self._client = httpx.AsyncClient(timeout=120.0)

    def register_server(self, model_key: str, base_url: str):
        """Register a running vLLM server."""
        self._servers[model_key] = base_url.rstrip("/")

    async def health_check(self, model_key: str) -> bool:
        """Check if a model server is healthy."""
        url = self._servers.get(model_key)
        if not url:
            return False
        try:
            resp = await self._client.get(f"{url}/health")
            return resp.status_code == 200
        except httpx.RequestError:
            return False

    async def generate(
        self,
        model_key: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> GenerationResult:
        """Send a non-streaming chat completion request."""
        url = f"{self._servers[model_key]}/v1/chat/completions"
        model_name = self._servers[model_key]  # vLLM uses the HF model name

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format

        start = time.perf_counter()
        resp = await self._client.post(url, json=payload)
        total_time = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            error_detail = resp.text[:500]
            # If tools caused the error, retry without tools
            if resp.status_code == 400 and tools and "tool" in error_detail.lower():
                payload.pop("tools", None)
                resp = await self._client.post(url, json=payload)
                total_time = (time.perf_counter() - start) * 1000
                if resp.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"vLLM error: {resp.text[:200]}", request=resp.request, response=resp
                    )
            else:
                raise httpx.HTTPStatusError(
                    f"vLLM error: {error_detail}", request=resp.request, response=resp
                )

        data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return GenerationResult(
            model_key=model_key,
            text=choice["message"]["content"] or "",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            ttft_ms=total_time,  # non-streaming, so TTFT ≈ total time
            total_time_ms=total_time,
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def generate_stream(
        self,
        model_key: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ):
        """
        Send a streaming chat completion request.
        Yields SSE chunks as they arrive, also tracks timing metrics.
        """
        import json

        url = f"{self._servers[model_key]}/v1/chat/completions"
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format

        start = time.perf_counter()
        first_token_time = None
        total_tokens = 0

        async with self._client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    yield {"type": "done", "ttft_ms": first_token_time, "total_tokens": total_tokens}
                    break

                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})

                if delta.get("content") and first_token_time is None:
                    first_token_time = (time.perf_counter() - start) * 1000

                if delta.get("content"):
                    total_tokens += 1

                yield {"type": "chunk", "data": data_str}

    async def close(self):
        await self._client.aclose()
