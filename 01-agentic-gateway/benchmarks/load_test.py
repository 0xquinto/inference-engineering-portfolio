"""
Load test for the Agentic Inference Gateway.

Simulates concurrent agent-style requests and measures:
- Routing accuracy (did complex requests go to the large model?)
- Latency under load (P50, P95, P99)
- Cost savings vs always-large-model
- Throughput (requests/second)

Usage:
    python benchmarks/load_test.py --url http://localhost:8080 --concurrency 10
    python benchmarks/load_test.py --url http://localhost:8080 --concurrency 50
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass

import httpx


# --- Test prompts: mix of simple and complex agent-style requests ---
SIMPLE_PROMPTS = [
    {"messages": [{"role": "user", "content": "What is 2 + 2?"}]},
    {"messages": [{"role": "user", "content": "Say hello in Spanish."}]},
    {"messages": [{"role": "user", "content": "What's the capital of France?"}]},
    {"messages": [{"role": "user", "content": "Convert 100 Fahrenheit to Celsius."}]},
    {"messages": [{"role": "user", "content": "Is Python a compiled language?"}]},
]

COMPLEX_PROMPTS = [
    {
        "messages": [{"role": "user", "content": "Analyze the time and space complexity of quicksort vs mergesort. Compare their performance characteristics in detail with examples."}],
    },
    {
        "messages": [{"role": "user", "content": "Write code to implement a rate limiter using the token bucket algorithm in Python. Include thread safety."}],
    },
    {
        "messages": [
            {"role": "user", "content": "Debug this: my async Python web server is leaking memory under load. What are the most likely causes and how do I diagnose them step by step?"},
        ],
    },
    {
        "messages": [{"role": "user", "content": "Design a distributed caching system. Explain the architecture, consistency model, and eviction strategies."}],
    },
]

TOOL_PROMPTS = [
    {
        "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            },
        }],
    },
    {
        "messages": [{"role": "user", "content": "Search the database for users who signed up in the last 7 days and summarize the trends."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Run a SQL query",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }],
    },
]


@dataclass
class RequestResult:
    prompt_type: str        # simple, complex, tool
    routed_to: str          # small or large
    complexity_score: float
    latency_ms: float
    status_code: int
    error: str | None = None


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    prompt_type: str,
) -> RequestResult:
    """Send a single request and collect metrics."""
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/v1/chat/completions",
            json={**payload, "max_tokens": 256},
        )
        latency = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            gateway = data.get("gateway", {})
            return RequestResult(
                prompt_type=prompt_type,
                routed_to=gateway.get("routed_to", "unknown"),
                complexity_score=gateway.get("complexity_score", 0),
                latency_ms=latency,
                status_code=200,
            )
        return RequestResult(
            prompt_type=prompt_type,
            routed_to="error",
            complexity_score=0,
            latency_ms=latency,
            status_code=resp.status_code,
            error=resp.text[:200],
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return RequestResult(
            prompt_type=prompt_type,
            routed_to="error",
            complexity_score=0,
            latency_ms=latency,
            status_code=0,
            error=str(e)[:200],
        )


async def run_load_test(url: str, concurrency: int, total_requests: int):
    """Run the load test with given concurrency."""
    # Build request queue: mix of simple, complex, and tool requests
    requests = []
    for i in range(total_requests):
        if i % 3 == 0:
            prompt = SIMPLE_PROMPTS[i % len(SIMPLE_PROMPTS)]
            requests.append((prompt, "simple"))
        elif i % 3 == 1:
            prompt = COMPLEX_PROMPTS[i % len(COMPLEX_PROMPTS)]
            requests.append((prompt, "complex"))
        else:
            prompt = TOOL_PROMPTS[i % len(TOOL_PROMPTS)]
            requests.append((prompt, "tool"))

    print(f"\nRunning load test: {total_requests} requests, concurrency={concurrency}")
    print(f"Target: {url}")
    print(f"Mix: {total_requests//3} simple, {total_requests//3} complex, {total_requests//3} tool\n")

    client = httpx.AsyncClient(timeout=120.0)
    results: list[RequestResult] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(payload, ptype):
        async with semaphore:
            return await send_request(client, url, payload, ptype)

    start = time.perf_counter()
    tasks = [bounded_request(p, t) for p, t in requests]
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start

    await client.aclose()

    # --- Report ---
    successful = [r for r in results if r.status_code == 200]
    failed = [r for r in results if r.status_code != 200]

    print("=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Total requests:  {len(results)}")
    print(f"Successful:      {len(successful)}")
    print(f"Failed:          {len(failed)}")
    print(f"Total time:      {total_time:.1f}s")
    print(f"Throughput:      {len(successful)/total_time:.1f} req/s")

    if successful:
        latencies = sorted([r.latency_ms for r in successful])
        print(f"\nLatency (ms):")
        print(f"  P50:  {latencies[len(latencies)//2]:.0f}")
        print(f"  P95:  {latencies[int(len(latencies)*0.95)]:.0f}")
        print(f"  P99:  {latencies[int(len(latencies)*0.99)]:.0f}")
        print(f"  Max:  {latencies[-1]:.0f}")

        # Routing analysis
        by_type = {}
        for r in successful:
            by_type.setdefault(r.prompt_type, []).append(r)

        print(f"\nRouting decisions:")
        for ptype, reqs in by_type.items():
            small = sum(1 for r in reqs if r.routed_to == "small")
            large = sum(1 for r in reqs if r.routed_to == "large")
            print(f"  {ptype:8s}: {small} → small, {large} → large")

    if failed:
        print(f"\nErrors:")
        for r in failed[:5]:
            print(f"  [{r.status_code}] {r.error}")

    # Fetch gateway metrics
    try:
        resp = httpx.get(f"{url}/metrics")
        if resp.status_code == 200:
            print(f"\nGateway metrics:")
            print(json.dumps(resp.json(), indent=2))
    except Exception:
        pass

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Load test the Agentic Gateway")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=60)
    args = parser.parse_args()

    asyncio.run(run_load_test(args.url, args.concurrency, args.requests))


if __name__ == "__main__":
    main()
