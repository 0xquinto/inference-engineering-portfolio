"""
Agentic Inference Gateway — FastAPI entrypoint.

A multi-model inference gateway with intelligent routing for agentic workloads.
Routes requests to small (8B) or large (70B) models based on complexity scoring.

Start:
    # Start vLLM servers first (see scripts/run_server.sh)
    python -m uvicorn src.main:app --host 0.0.0.0 --port 8080
"""

import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .router.classifier import ComplexityClassifier
from .router.strategy import RoutingStrategy
from .serving.engine import InferenceEngine
from .serving.models import ModelRegistry
from .serving.streaming import stream_response
from .dashboard.metrics import MetricsCollector, RequestMetric
from .agents.tool_parser import ToolCallParser

# --- Configuration ---
SMALL_MODEL_URL = os.getenv("SMALL_MODEL_URL", "http://localhost:8001")
LARGE_MODEL_URL = os.getenv("LARGE_MODEL_URL", "http://localhost:8002")
CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/models.yaml")
COMPLEXITY_THRESHOLD = float(os.getenv("COMPLEXITY_THRESHOLD", "0.6"))

# --- Global state ---
engine = InferenceEngine()
registry = ModelRegistry(CONFIG_PATH)
classifier = ComplexityClassifier(threshold=COMPLEXITY_THRESHOLD)
metrics = MetricsCollector()
tool_parser = ToolCallParser()
strategy: RoutingStrategy | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global strategy

    # Register model servers
    engine.register_server("small", SMALL_MODEL_URL)
    engine.register_server("large", LARGE_MODEL_URL)

    # Check which models are actually available
    available = []
    for key in ["small", "large"]:
        if await engine.health_check(key):
            available.append(key)
            print(f"  [ok] {key} model at {engine._servers[key]}")
        else:
            print(f"  [!!] {key} model not available at {engine._servers[key]}")

    if not available:
        print("WARNING: No model servers available. Start vLLM servers first.")
        print("  See: scripts/run_server.sh")
        available = ["small"]  # allow startup for testing

    strategy = RoutingStrategy(classifier, available)
    print(f"Gateway ready. Available models: {available}")

    yield

    await engine.close()


app = FastAPI(
    title="Agentic Inference Gateway",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    statuses = {}
    for key in ["small", "large"]:
        statuses[key] = await engine.health_check(key)
    return {"status": "ok", "models": statuses}


@app.get("/metrics")
async def get_metrics():
    """Returns routing stats, cost savings, and latency percentiles."""
    return metrics.summary()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    messages = body.get("messages", [])
    tools = body.get("tools")
    response_format = body.get("response_format")
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 1024)
    temperature = body.get("temperature", 0.7)
    requested_model = body.get("model")

    # Route the request
    model_key, reason = strategy.select_model(
        messages=messages,
        requested_model=requested_model,
        tools=tools,
        response_format=response_format,
    )

    complexity = classifier.classify(messages, tools, response_format)

    if stream:
        # Streaming response
        async def event_generator():
            start = time.perf_counter()
            token_count = 0
            async for chunk in stream_response(
                engine, model_key, messages, max_tokens, temperature, tools, response_format
            ):
                if "data: [DONE]" not in chunk:
                    token_count += 1
                yield chunk
            total_ms = (time.perf_counter() - start) * 1000

            # Record metrics after stream completes
            cost_config = registry.get_cost(model_key)
            metrics.record(RequestMetric(
                timestamp=time.time(),
                model_key=model_key,
                complexity_score=complexity.score,
                input_tokens=0,  # not available in streaming
                output_tokens=token_count,
                ttft_ms=0,
                total_time_ms=total_ms,
                had_tool_calls=False,
                cost_usd=cost_config.estimate_cost(0, token_count),
            ))

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "X-Gateway-Model": model_key,
                "X-Gateway-Reason": reason,
                "X-Complexity-Score": str(round(complexity.score, 3)),
            },
        )

    # Non-streaming response
    start = time.perf_counter()
    result = await engine.generate(
        model_key=model_key,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        response_format=response_format,
    )
    total_ms = (time.perf_counter() - start) * 1000

    # Track cost
    cost_config = registry.get_cost(model_key)
    cost = cost_config.estimate_cost(result.input_tokens, result.output_tokens)

    # Record metrics
    metrics.record(RequestMetric(
        timestamp=time.time(),
        model_key=model_key,
        complexity_score=complexity.score,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        ttft_ms=result.ttft_ms,
        total_time_ms=total_ms,
        had_tool_calls=False,
        cost_usd=cost,
    ))

    # Return OpenAI-compatible response with gateway metadata
    return JSONResponse(
        content={
            "id": f"gateway-{int(time.time()*1000)}",
            "object": "chat.completion",
            "model": registry.get(model_key).name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }],
            "usage": {
                "prompt_tokens": result.input_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
            },
            "gateway": {
                "routed_to": model_key,
                "reason": reason,
                "complexity_score": round(complexity.score, 3),
                "cost_usd": round(cost, 8),
                "latency_ms": round(total_ms, 2),
            },
        },
        headers={
            "X-Gateway-Model": model_key,
            "X-Gateway-Reason": reason,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
