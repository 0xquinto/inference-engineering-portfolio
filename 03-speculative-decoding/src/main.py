"""
Speculative Decoding & MoE Experiments — entrypoint.

Usage:
    # Part A: Speculative decoding with vLLM's built-in support
    python src/main.py speculative --config configs/speculative.yaml

    # Part B: MoE expert profiling
    python src/main.py moe --model mistralai/Mixtral-8x7B-Instruct-v0.1
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding & MoE Experiments")
    subparsers = parser.add_subparsers(dest="command")

    # Part A
    spec_parser = subparsers.add_parser("speculative")
    spec_parser.add_argument("--config", default="configs/speculative.yaml")

    # Part B
    moe_parser = subparsers.add_parser("moe")
    moe_parser.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1")

    args = parser.parse_args()

    if args.command == "speculative":
        # TODO: Step 1 — Load experiment configs
        # TODO: Step 2 — For each draft/target pair:
        #   a. Start vLLM with --speculative-model flag
        #   b. Run standard decoding baseline
        #   c. Run speculative decoding with each num_speculative_tokens value
        #   d. Collect acceptance rates, speedup, GPU utilization
        # TODO: Step 3 — Generate comparison plots
        print("Running speculative decoding experiments...")

    elif args.command == "moe":
        # TODO: Step 1 — Load MoE model with hooks into expert layers
        # TODO: Step 2 — Run diverse prompts and log expert activations
        # TODO: Step 3 — Generate expert activation heatmaps
        # TODO: Step 4 — Analyze hot vs cold experts
        print("Running MoE expert profiling...")


if __name__ == "__main__":
    main()
