"""
Inference Stack Benchmark Suite — CLI entrypoint.

Usage:
    python -m src.main --engine vllm --config configs/engines.yaml
    python -m src.main --engine all --config configs/engines.yaml
    python -m src.main --list-engines
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import yaml

from .runners import VllmRunner, SglangRunner, TrtllmRunner
from .metrics import LatencyTracker, ThroughputCalculator, get_gpu_memory

RUNNERS = {
    "vllm": VllmRunner,
    "sglang": SglangRunner,
    "tensorrt-llm": TrtllmRunner,
}


def load_prompts(path: str = "configs/prompts.json") -> dict[str, list[dict]]:
    with open(path) as f:
        return json.load(f)


def load_config(path: str = "configs/engines.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def run_engine_benchmark(
    engine_name: str,
    model: str,
    prompts: dict[str, list[dict]],
    concurrency_levels: list[int],
    config: dict,
) -> list[dict]:
    """Run full benchmark suite for a single engine."""
    engine_cfg = config["engines"][engine_name.replace("-", "_")]
    runner_cls = RUNNERS[engine_name]
    runner = runner_cls(port=engine_cfg["port"])

    print(f"\n{'='*60}")
    print(f"Benchmarking: {engine_name}")
    print(f"{'='*60}")

    extra_args = engine_cfg.get("extra_args", [])
    await runner.start_server(model, extra_args=extra_args)

    # Get baseline GPU memory
    mem_used, mem_total = get_gpu_memory()
    print(f"GPU Memory: {mem_used} / {mem_total} MB")

    all_results = []
    bench_cfg = config.get("benchmark", {})

    for category, prompt_list in prompts.items():
        flat_prompts = [p["prompt"] for p in prompt_list]

        for concurrency in concurrency_levels:
            print(f"  [{category}] concurrency={concurrency}...", end=" ", flush=True)
            latency = LatencyTracker()
            throughput = ThroughputCalculator()

            # Warmup
            for p in flat_prompts[:2]:
                try:
                    await runner.send_request(p)
                except Exception:
                    pass

            # Benchmark
            results = await runner.run_concurrent(
                flat_prompts * bench_cfg.get("requests_per_prompt", 1),
                concurrency,
            )

            for r in results:
                r.prompt_category = category
                latency.record(r.total_time_ms)
                throughput.record_request(r.total_time_ms, r.tokens_generated)

            lat_summary = latency.summary()
            thr_summary = throughput.summary()
            mem_used, _ = get_gpu_memory()

            result_entry = {
                "engine": engine_name,
                "category": category,
                "concurrency": concurrency,
                "latency": lat_summary,
                "throughput": thr_summary,
                "gpu_memory_mb": mem_used,
            }
            all_results.append(result_entry)
            print(f"p50={lat_summary['p50']}ms, {thr_summary['avg_tokens_per_sec']} tok/s")

    await runner.stop_server()
    return all_results


async def async_main(args):
    config = load_config(args.config)
    prompts = load_prompts(args.prompts)
    model = config["model"]["name"]
    concurrency_levels = config["benchmark"]["concurrency_levels"]

    engines = list(RUNNERS.keys()) if args.engine == "all" else [args.engine]
    all_results = {}

    for engine in engines:
        results = await run_engine_benchmark(engine, model, prompts, concurrency_levels, config)
        all_results[engine] = results

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    output_path = output_dir / f"benchmark_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Inference Stack Benchmarks")
    parser.add_argument("--engine", choices=["vllm", "sglang", "tensorrt-llm", "all"], default="all")
    parser.add_argument("--config", default="configs/engines.yaml")
    parser.add_argument("--prompts", default="configs/prompts.json")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--list-engines", action="store_true", help="List available engines")
    args = parser.parse_args()

    if args.list_engines:
        for name in RUNNERS:
            print(name)
        return

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
