import argparse
import asyncio
import json
from pathlib import Path

from .config import load_config
from .schemas import get_schema, get_prompt


def main():
    parser = argparse.ArgumentParser(description="Structured Output & Constrained Decoding Benchmarks")
    parser.add_argument("--config", default="configs/structured.yaml", help="Config file path")
    parser.add_argument("--profile", choices=["gpu", "local"], help="Hardware profile (gpu or local)")
    parser.add_argument("--output", default="results/", help="Output directory for results")
    parser.add_argument("--list-backends", action="store_true", help="List available backends")
    parser.add_argument("--list-schemas", action="store_true", help="List schema complexity levels")
    parser.add_argument("--backend", help="Run only this backend (default: all)")
    parser.add_argument("--schema", help="Run only this schema level (default: all)")
    parser.add_argument("--step", choices=["benchmark", "visualize", "all"],
                        default="all", help="Which pipeline step to run")
    args = parser.parse_args()

    if args.profile:
        config_path = Path(f"profiles/{args.profile}.yaml")
    else:
        config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.list_backends:
        print("Available backends:")
        for b in cfg.backends:
            constrained = "" if b.is_constrained else " (unconstrained)"
            print(f"  {b.name}: {b.description}{constrained}")
        return

    if args.list_schemas:
        print("Available schema levels:")
        for s in cfg.schemas:
            print(f"  {s.name}: {s.description} (complexity: {s.complexity})")
        return

    asyncio.run(async_main(args, cfg))


async def async_main(args, cfg):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    backends = cfg.backends
    if args.backend:
        backends = [b for b in backends if b.name == args.backend]
        if not backends:
            print(f"Error: backend '{args.backend}' not found")
            return

    schemas = cfg.schemas
    if args.schema:
        schemas = [s for s in schemas if s.name == args.schema]
        if not schemas:
            print(f"Error: schema '{args.schema}' not found")
            return

    all_results = {}

    if args.step in ("benchmark", "all"):
        print("=== Step 1: Structured Output Benchmarks ===")
        from .benchmark import StructuredBenchmarker
        benchmarker = StructuredBenchmarker(
            port=cfg.port, max_tokens=cfg.max_tokens,
            temperature=cfg.temperature, max_retries=cfg.max_retries,
            model_name=cfg.model_id,
        )

        for backend in backends:
            print(f"\n  Backend: {backend.name}")
            for schema_level in schemas:
                print(f"    Schema: {schema_level.name} ({schema_level.complexity})")
                schema_dict = get_schema(schema_level.name) if backend.is_constrained else None
                prompt = get_prompt(schema_level.name)

                for conc in cfg.concurrency_levels:
                    prompts = [prompt] * cfg.requests_per_prompt
                    results = await benchmarker.run_concurrent(
                        prompts, conc, schema_dict, backend.name, schema_level.name,
                    )
                    agg = benchmarker.aggregate(results)
                    backend_data = all_results.setdefault(backend.name, {})
                    schema_data = backend_data.setdefault(schema_level.name, {})
                    schema_data["ttft_p50"] = agg.get("ttft_p50", 0)
                    schema_data["throughput_p50"] = agg.get("throughput_p50", 0)
                    schema_data["validity_rate"] = agg.get("validity_rate", 0)
                    schema_data["avg_retries"] = agg.get("avg_retries", 0)
                    print(
                        f"      c={conc}: TTFT={agg.get('ttft_p50', 0):.1f}ms, "
                        f"TPS={agg.get('throughput_p50', 0):.1f}, "
                        f"valid={agg.get('validity_rate', 0):.0%}"
                    )

    if args.step in ("visualize", "all"):
        print("\n=== Step 2: Visualization ===")
        from .visualize import plot_tps_overhead, plot_validity_rate, plot_latency_comparison
        plot_tps_overhead(all_results, str(output))
        plot_validity_rate(all_results, str(output))
        plot_latency_comparison(all_results, str(output))
        print(f"  Charts saved to {output}/")

    results_file = output / "structured_output_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
