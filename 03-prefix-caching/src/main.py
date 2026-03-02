import argparse
import asyncio
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Prefix Caching & KV Cache Optimization Benchmarks")
    parser.add_argument("--config", default="configs/scenarios.yaml", help="Scenario config")
    parser.add_argument("--engines", default="configs/engines.yaml", help="Engine config")
    parser.add_argument("--output", default="results/", help="Output directory")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    parser.add_argument("--scenario", help="Run only this scenario")
    parser.add_argument("--engine", choices=["vllm", "sglang"], default="vllm", help="Engine to benchmark")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.engines) as f:
        engines = yaml.safe_load(f)

    if args.list_scenarios:
        print("Available scenarios:")
        for name, data in config["scenarios"].items():
            print(f"  {name}: {data['description']}")
        return

    asyncio.run(async_main(args, config, engines))


async def async_main(args, config, engines):
    from .benchmark import CacheBenchmarker
    from .metrics import CacheMetrics
    from .scenarios import (
        generate_shared_system_prompt,
        generate_multi_turn,
        generate_rag_context,
        generate_cache_pressure,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    engine_cfg = engines["engines"][args.engine]
    port = engine_cfg["port"]
    bench_cfg = config["benchmark"]
    benchmarker = CacheBenchmarker(port=port, max_tokens=bench_cfg["max_tokens"])
    metrics = CacheMetrics()
    all_summaries = {}

    scenarios = config["scenarios"]
    if args.scenario:
        scenarios = {k: v for k, v in scenarios.items() if k == args.scenario}

    for scenario_name, scenario_cfg in scenarios.items():
        print(f"\n=== Scenario: {scenario_name} ===")
        print(f"  {scenario_cfg['description']}")

        for caching in [False, True]:
            label = "ON" if caching else "OFF"
            print(f"  Running with prefix caching {label}...")

            if scenario_name == "shared_system_prompt":
                requests = generate_shared_system_prompt(
                    num_requests=scenario_cfg["num_requests"],
                    system_tokens=scenario_cfg["system_prompt_tokens"],
                    user_tokens=scenario_cfg["user_message_tokens"],
                )
                results = await benchmarker.run_scenario_shared_system(
                    requests, scenario_name, caching, bench_cfg["concurrency"]
                )
            elif scenario_name == "multi_turn":
                convos = generate_multi_turn(
                    num_conversations=scenario_cfg["num_conversations"],
                    turns=scenario_cfg["turns_per_conversation"],
                    tokens_per_turn=scenario_cfg["tokens_per_turn"],
                )
                results = await benchmarker.run_scenario_multi_turn(
                    convos, scenario_name, caching
                )
            elif scenario_name == "rag_common_context":
                requests = generate_rag_context(
                    context_tokens=scenario_cfg["context_tokens"],
                    num_queries=scenario_cfg["num_queries"],
                    query_tokens=scenario_cfg["query_tokens"],
                )
                results = await benchmarker.run_scenario_shared_system(
                    requests, scenario_name, caching, bench_cfg["concurrency"]
                )
            elif scenario_name == "cache_pressure":
                batches = generate_cache_pressure(
                    prefix_tokens=scenario_cfg["prefix_tokens"],
                    unique_prefixes=scenario_cfg["unique_prefixes"],
                    requests_per_prefix=scenario_cfg["requests_per_prefix"],
                )
                pressure_results = await benchmarker.run_scenario_cache_pressure(
                    batches, scenario_name, caching, bench_cfg["concurrency"]
                )
                results = []
                for prefix_results in pressure_results.values():
                    results.extend(prefix_results)
            else:
                print(f"  Unknown scenario: {scenario_name}, skipping")
                continue

            for r in results:
                metrics.record(r)

            summary = metrics.summary(scenario_name, caching)
            cache_key = "caching_on" if caching else "caching_off"
            all_summaries.setdefault(scenario_name, {})[cache_key] = summary
            print(f"    TTFT p50: {summary.get('ttft_p50', 0):.1f}ms ({summary['count']} requests)")

        speedup = metrics.ttft_speedup(scenario_name)
        print(f"  Speedup: {speedup:.1f}x")

    results_file = output / "prefix_caching_results.json"
    with open(results_file, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    from .visualize import plot_ttft_comparison, plot_cache_pressure
    plot_ttft_comparison(all_summaries, str(output))
    print(f"Charts saved to {output}/")


if __name__ == "__main__":
    main()
