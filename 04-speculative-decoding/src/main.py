import argparse
import asyncio
import json
from pathlib import Path

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Benchmarks for vLLM")
    parser.add_argument("--config", default="configs/speculative.yaml", help="Config file path")
    parser.add_argument("--profile", choices=["gpu", "local"],
                        help="Hardware profile (gpu or local)")
    parser.add_argument("--output", default="results/", help="Output directory for results")
    parser.add_argument("--list-methods", action="store_true", help="List available speculative methods")
    parser.add_argument("--method", help="Run only this method (default: all)")
    parser.add_argument("--step", choices=["benchmark", "visualize", "all"],
                        default="all", help="Which pipeline step to run")
    args = parser.parse_args()

    if args.profile:
        config_path = Path(f"profiles/{args.profile}.yaml")
    else:
        config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.list_methods:
        print("Available speculative decoding methods:")
        for method in cfg.methods:
            baseline = " (baseline)" if method.is_baseline else ""
            print(f"  {method.name}: {method.description}{baseline}")
        return

    asyncio.run(async_main(args, cfg))


async def async_main(args, cfg):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    methods = cfg.methods
    if args.method:
        methods = [m for m in methods if m.name == args.method]
        if not methods:
            print(f"Error: method '{args.method}' not found")
            return

    all_results = {}

    if args.step in ("benchmark", "all"):
        print("=== Step 1: Speculative Decoding Benchmarks ===")
        from .benchmark import SpecBenchmarker
        from .methods import SpecMethodTracker

        benchmarker = SpecBenchmarker(port=cfg.port, max_tokens=cfg.max_tokens, temperature=cfg.temperature)
        tracker = SpecMethodTracker()

        for method in methods:
            print(f"  Benchmarking {method.name}...")
            for qps in cfg.qps_levels:
                prompts = cfg.benchmark_prompts * cfg.requests_per_prompt
                results = await benchmarker.run_at_qps(prompts, qps, method.name)
                for r in results:
                    tracker.record(r)
                agg = benchmarker.aggregate(results)
                method_data = all_results.setdefault(method.name, {})
                method_data[str(qps)] = {
                    "ttft_p50": agg.get("ttft_p50", 0),
                    "ttft_p95": agg.get("ttft_p95", 0),
                    "throughput_p50": agg.get("throughput_p50", 0),
                    "count": agg.get("count", 0),
                }
                print(f"    qps={qps}: TTFT={agg.get('ttft_p50', 0):.1f}ms, TPS={agg.get('throughput_p50', 0):.1f}")

        print("\n  Speedups vs baseline:")
        for method in methods:
            if method.is_baseline:
                continue
            for qps in cfg.qps_levels:
                spd = tracker.speedup(method.name, qps)
                print(f"    {method.name} @ qps={qps}: {spd:.2f}x")

    if args.step in ("visualize", "all"):
        print("\n=== Step 2: Visualization ===")
        from .visualize import plot_ttft_by_qps, plot_throughput_by_qps, plot_speedup_heatmap
        plot_ttft_by_qps(all_results, str(output))
        plot_throughput_by_qps(all_results, str(output))
        plot_speedup_heatmap(all_results, str(output))
        print(f"  Charts saved to {output}/")

    results_file = output / "speculative_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
