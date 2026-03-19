import argparse
import asyncio
import json
from pathlib import Path

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="LLM Cascade & Cost Optimization Analysis")
    parser.add_argument("--config", default="configs/cost.yaml", help="Config file path")
    parser.add_argument("--output", default="results/", help="Output directory for results")
    parser.add_argument("--profile", choices=["gpu", "local"], help="Hardware profile (gpu or local)")
    parser.add_argument("--list-models", action="store_true", help="List available model tiers")
    parser.add_argument("--step", choices=["benchmark", "analyze", "visualize", "all"],
                        default="all", help="Which pipeline step to run")
    args = parser.parse_args()

    if args.profile:
        config_path = Path(f"profiles/{args.profile}.yaml")
    else:
        config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.list_models:
        print("Available model tiers:")
        for model in cfg.models:
            print(f"  {model.name}: {model.model_name} ({model.params}) - {model.description}")
        return

    asyncio.run(async_main(args, cfg))


async def async_main(args, cfg):
    from .cascade import CascadeRouter
    from .cost_model import (
        build_cost_table,
        compare_with_apis,
        cascade_cost_estimate,
    )
    from .visualize import (
        plot_cost_per_token,
        plot_cascade_distribution,
        plot_quality_vs_cost,
        plot_utilization_breakeven,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    router = CascadeRouter(cfg.cascade, cfg.models)
    all_results = {}

    if args.step in ("benchmark", "all"):
        print("=== Step 1: Cascade Benchmark ===")
        from .benchmark import CascadeBenchmarker

        model_ports = {m.name: m.port for m in cfg.models}
        model_ids = {m.name: m.model_id for m in cfg.models}
        benchmarker = CascadeBenchmarker(
            models=model_ports,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            model_ids=model_ids,
        )

        results = await benchmarker.run_cascade(
            cfg.prompts, router, cfg.concurrency
        )
        agg = CascadeBenchmarker.aggregate(results)
        all_results["benchmark"] = {
            "aggregate": agg,
            "raw": [r.to_dict() for r in results],
        }
        print(f"  {agg['count']} requests completed")
        for model_name, model_agg in agg.get("per_model", {}).items():
            print(f"  {model_name}: TTFT={model_agg['ttft_p50']:.1f}ms, TPS={model_agg['throughput_p50']:.1f}")

    if args.step in ("analyze", "all"):
        print("\n=== Step 2: Cost Analysis ===")
        # Use benchmark data if available, otherwise use placeholder TPS values
        tps_by_model = {}
        bench_data = all_results.get("benchmark", {}).get("aggregate", {})
        per_model = bench_data.get("per_model", {})
        if per_model:
            for model_name, model_agg in per_model.items():
                tps_by_model[model_name] = model_agg.get("throughput_p50", 0)
        else:
            # Placeholder values for analysis-only mode
            tps_by_model = {"small": 250.0, "medium": 85.0, "large": 25.0}

        cost_table = build_cost_table(
            cfg.models, tps_by_model, cfg.cost_analysis.target_utilization
        )

        api_cmp = compare_with_apis(cost_table, cfg.cost_analysis.api_comparison)

        # Build cascade decisions from prompts
        decisions = []
        for complexity, prompts in cfg.prompts.items():
            for prompt in prompts:
                decisions.append(router.route(prompt))

        cascade_cost = cascade_cost_estimate(decisions, cost_table)

        all_results["cost_analysis"] = {
            "cost_table": [
                {
                    "model": e.model_name,
                    "tps": e.tokens_per_second,
                    "gpu_cost_per_hour": e.gpu_cost_per_hour,
                    "cost_per_million": e.cost_per_million_tokens,
                    "monthly_cost": e.monthly_cost_at_utilization,
                    "monthly_tokens": e.monthly_token_capacity,
                }
                for e in cost_table
            ],
            "api_comparison": api_cmp,
            "cascade_cost": cascade_cost,
        }

        for e in cost_table:
            print(f"  {e.model_name}: ${e.cost_per_million_tokens:.2f}/M tokens @ {e.tokens_per_second:.0f} tps")
        print(f"  Cascade blended: ${cascade_cost['blended_cost_per_million']:.2f}/M tokens")

    if args.step in ("visualize", "all"):
        print("\n=== Step 3: Visualization ===")
        cost_data = all_results.get("cost_analysis", {})
        cost_table = cost_data.get("cost_table", [])
        api_prices = cfg.cost_analysis.api_comparison

        if cost_table:
            plot_cost_per_token(cost_table, api_prices, str(output))
            plot_utilization_breakeven(cost_table, api_prices, str(output))
            print(f"  Cost charts saved to {output}/")

        # Build decisions for distribution chart
        decisions_dicts = []
        for complexity, prompts in cfg.prompts.items():
            for prompt in prompts:
                d = router.route(prompt)
                decisions_dicts.append({
                    "prompt": d.prompt,
                    "routed_to": d.routed_to,
                    "final_model": d.final_model,
                })

        if decisions_dicts:
            plot_cascade_distribution(decisions_dicts, str(output))
            print(f"  Distribution chart saved to {output}/")

        # Quality vs cost scatter (needs both benchmark + cost data)
        quality_data = []
        for entry in cost_table:
            quality_data.append({
                "model": entry["model"],
                "cost_per_million": entry["cost_per_million"],
                "quality_score": 0.0,
            })
        if quality_data:
            plot_quality_vs_cost(quality_data, str(output))
            print(f"  Quality chart saved to {output}/")

    results_file = output / "cost_optimization_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
