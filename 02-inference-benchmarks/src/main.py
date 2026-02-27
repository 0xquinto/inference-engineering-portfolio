"""
Inference Stack Benchmark Suite — CLI entrypoint.

Usage:
    python src/main.py --engine vllm --model meta-llama/Llama-3.1-8B-Instruct
    python src/main.py --engine sglang --model meta-llama/Llama-3.1-8B-Instruct
    python src/main.py --engine tgi --model meta-llama/Llama-3.1-8B-Instruct
    python src/main.py --all  # Run all engines and generate comparison
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Inference Stack Benchmarks")
    parser.add_argument("--engine", choices=["vllm", "sglang", "tgi", "all"])
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 10, 50, 100])
    parser.add_argument("--quantization", choices=["none", "awq"], default="none")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    # TODO: Step 1 — Load prompts from configs/prompts.json
    # TODO: Step 2 — Initialize the appropriate runner(s)
    # TODO: Step 3 — Run benchmarks across concurrency levels
    # TODO: Step 4 — Save raw results as JSON
    # TODO: Step 5 — Generate comparison plots
    print(f"Running {args.engine} benchmark with {args.model}")


if __name__ == "__main__":
    main()
