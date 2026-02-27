"""
SSE streaming utilities.

Handles Server-Sent Events streaming from vLLM to the client,
with proper formatting and error handling.
"""

import json
from collections.abc import AsyncIterator


async def stream_response(
    engine,
    model_key: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    tools: list[dict] | None = None,
    response_format: dict | None = None,
) -> AsyncIterator[str]:
    """
    Stream SSE chunks from vLLM to the client.

    Yields properly formatted SSE lines: "data: {json}\n\n"
    """
    async for event in engine.generate_stream(
        model_key=model_key,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        response_format=response_format,
    ):
        if event["type"] == "chunk":
            yield f"data: {event['data']}\n\n"
        elif event["type"] == "done":
            yield "data: [DONE]\n\n"
