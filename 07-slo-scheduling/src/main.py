import argparse
import asyncio
import json
from pathlib import Path

from .config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="SLO-Aware Request Scheduling Benchmarks",
    )
    parser.add_argument("--config", default="configs/scheduling.yaml",
                        help="Config file path")
    parser.add_argument("--profile", choices=["gpu", "local"],
                        help="Hardware profile (gpu or local)")
    parser.add_argument("--output", default="results/",
                        help="Output directory for results")
    parser.add_argument("--policy",
                        help="Run only this policy (default: all from config)")
    parser.add_argument("--list-policies", action="store_true",
                        help="List available scheduling policies")
    parser.add_argument("--step", choices=["benchmark", "visualize", "all"],
                        default="all", help="Which pipeline step to run")
    args = parser.parse_args()

    if args.profile:
        config_path = Path(f"profiles/{args.profile}.yaml")
    else:
        config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.list_policies:
        print("Available policies:")
        for p in cfg.policies:
            print(f"  {p}")
        return

    asyncio.run(async_main(args, cfg))


async def async_main(args, cfg):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    results_file = output / "scheduling_results.json"

    policies = cfg.policies
    if args.policy:
        if args.policy not in policies:
            print(f"Error: policy '{args.policy}' not in config")
            return
        policies = [args.policy]

    all_results = {}
    if results_file.exists():
        try:
            with open(results_file) as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass

    if args.step in ("benchmark", "all"):
        print("=== Step 1: Scheduling Benchmarks ===")
        from .benchmark import SchedulingBenchmarker
        from .workload import WorkloadGenerator
        from .metrics import compute_goodput, compute_goodput_per_class, compute_fairness, compute_latency_percentiles

        for policy in policies:
            print(f"\n  Policy: {policy}")
            all_results[policy] = {}

            benchmarker = SchedulingBenchmarker(
                port=cfg.port, max_tokens=cfg.max_tokens,
                temperature=cfg.temperature, model_name=cfg.model_id,
                policy=policy,
                scheduler_config={
                    "max_queue_depth": cfg.scheduler.max_queue_depth,
                    "max_concurrent": cfg.scheduler.max_concurrent,
                },
            )

            gen = WorkloadGenerator(cfg.workload_classes, seed=42)

            for qps in cfg.qps_levels:
                requests = gen.generate(cfg.requests_per_qps)
                print(f"    QPS={qps} ({len(requests)} requests)...")

                results = await benchmarker.run_at_qps(requests, qps)

                per_class = compute_goodput_per_class(results)
                overall = compute_goodput(results)
                fairness = compute_fairness(per_class)
                latencies = [r.latency_seconds for r in results]
                ttfts = [r.ttft_seconds for r in results]
                lat_pct = compute_latency_percentiles(latencies)
                ttft_pct = compute_latency_percentiles(ttfts)

                all_results[policy][qps] = {
                    "goodput": overall,
                    "goodput_per_class": per_class,
                    "fairness": fairness,
                    "latency_p50": lat_pct["p50"],
                    "latency_p95": lat_pct["p95"],
                    "ttft_p50": ttft_pct["p50"],
                    "ttft_p95": ttft_pct["p95"],
                    "latencies": latencies,
                    "total_requests": len(requests),
                }

                print(
                    f"      goodput={overall:.0%}, fairness={fairness:.2f}, "
                    f"lat_p50={lat_pct['p50']:.1f}s, lat_p95={lat_pct['p95']:.1f}s"
                )

    if args.step in ("visualize", "all"):
        print("\n=== Step 2: Visualization ===")
        if not all_results and results_file.exists():
            with open(results_file) as f:
                all_results = json.load(f)
        from .visualize import plot_goodput_vs_qps, plot_latency_cdf, plot_fairness_heatmap
        slos = {wc.name: wc.slo_seconds for wc in cfg.workload_classes}
        plot_goodput_vs_qps(all_results, str(output))
        plot_latency_cdf(all_results, slos, str(output))
        plot_fairness_heatmap(all_results, str(output))
        print(f"  Charts saved to {output}/")

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
