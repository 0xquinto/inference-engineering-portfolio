import argparse
import asyncio
import json
from pathlib import Path

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Quantization Pipeline & Quality-Speed Tradeoffs")
    parser.add_argument("--config", default="configs/quantization.yaml", help="Config file path")
    parser.add_argument("--output", default="results/", help="Output directory for results")
    parser.add_argument("--list-formats", action="store_true", help="List available quantization formats")
    parser.add_argument("--step", choices=["quantize", "evaluate", "benchmark", "visualize", "all"],
                        default="all", help="Which pipeline step to run")
    parser.add_argument("--format", help="Run only this format (default: all)")
    parser.add_argument("--profile", choices=["gpu", "local"], help="Hardware profile (gpu or local)")
    args = parser.parse_args()

    if args.profile:
        config_path = Path(f"profiles/{args.profile}.yaml")
    else:
        config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.list_formats:
        print("Available quantization formats:")
        for fmt in cfg.formats:
            baseline = " (baseline)" if fmt.is_baseline else ""
            print(f"  {fmt.name}: {fmt.description}{baseline}")
        return

    asyncio.run(async_main(args, cfg))


async def async_main(args, cfg):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    formats = cfg.formats
    if args.format:
        formats = [f for f in formats if f.name == args.format]
        if not formats:
            print(f"Error: format '{args.format}' not found")
            return

    all_results = {}

    if args.step in ("quantize", "all"):
        print("=== Step 1: Quantization ===")
        from .quantize import QuantizationRunner
        runner = QuantizationRunner(model_name=cfg.model_name)
        for fmt in formats:
            if fmt.is_baseline:
                print(f"  Skipping {fmt.name} (baseline)")
                continue
            print(f"  Quantizing {fmt.name}...")
            result = runner.quantize(fmt)
            print(f"  Done in {result.time_seconds:.1f}s, compression {result.compression_ratio:.1f}x")

    if args.step in ("evaluate", "all"):
        print("\n=== Step 2: Quality Evaluation ===")
        from .evaluate import QualityEvaluator
        for fmt in formats:
            model_path = fmt.vllm_model_path(cfg.model_name)
            print(f"  Evaluating {fmt.name} ({model_path})...")
            evaluator = QualityEvaluator(model_path=model_path, max_samples=cfg.perplexity_max_samples)
            eval_result = evaluator.evaluate(fmt.name)
            all_results.setdefault(fmt.name, {}).update(eval_result.to_dict())
            print(f"  Perplexity: {eval_result.perplexity:.2f}, MMLU: {eval_result.mmlu_accuracy:.3f}")

    if args.step in ("benchmark", "all"):
        print("\n=== Step 3: Performance Benchmarks ===")
        from .benchmark import PerfBenchmarker
        benchmarker = PerfBenchmarker(port=cfg.engine_port, max_tokens=cfg.max_tokens)
        for fmt in formats:
            print(f"  Benchmarking {fmt.name}...")
            for conc in cfg.concurrency_levels:
                prompts = cfg.benchmark_prompts * cfg.requests_per_prompt
                results = await benchmarker.run_concurrent(prompts, conc, fmt.name)
                agg = benchmarker.aggregate(results)
                fmt_data = all_results.setdefault(fmt.name, {})
                fmt_data.setdefault("ttft_ms", {})[str(conc)] = agg.get("ttft_p50", 0)
                fmt_data.setdefault("throughput_tps", {})[str(conc)] = agg.get("throughput_p50", 0)
                fmt_data["vram_mb"] = agg.get("avg_gpu_memory_mb", 0)
                print(f"    c={conc}: TTFT={agg.get('ttft_p50', 0):.1f}ms, TPS={agg.get('throughput_p50', 0):.1f}")

    if args.step in ("visualize", "all"):
        print("\n=== Step 4: Visualization ===")
        from .visualize import plot_comparison, plot_pareto
        plot_comparison(all_results, str(output))
        plot_pareto(all_results, str(output))
        print(f"  Charts saved to {output}/")

    results_file = output / "quantization_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
