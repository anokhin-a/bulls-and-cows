#!/usr/bin/env python3
"""
Generate pre-sampled JSONL datasets for each difficulty level (1-10).

Outputs:
    data/difficulty_{d}.jsonl   for d in 1..10

Each file contains `num_per_difficulty` puzzles with a fixed random seed
for full reproducibility.
"""

import argparse
import json
import os
import random
import time

from bulls_and_cows.bulls_and_cows import BullsCowsEnv


def main():
    parser = argparse.ArgumentParser(
        description="Generate Bulls-and-Cows datasets for each difficulty."
    )
    parser.add_argument(
        "--num", type=int, default=100,
        help="Number of puzzles per difficulty level (default: 100)",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=300,
        help="Max attempts to generate one puzzle (default: 300)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--outdir", type=str, default="data",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--difficulties", type=str, default="1-10",
        help="Difficulty range, e.g. '1-10' or '1-5' (default: 1-10)",
    )
    args = parser.parse_args()

    # Parse difficulty range
    lo, hi = map(int, args.difficulties.split("-"))
    difficulties = list(range(lo, hi + 1))

    os.makedirs(args.outdir, exist_ok=True)

    env = BullsCowsEnv()

    for diff in difficulties:
        # Set seed per difficulty for reproducibility
        seed = args.seed + diff
        random.seed(seed)

        print(f"[difficulty={diff:>2}] generating {args.num} puzzles (seed={seed})...")
        t0 = time.time()
        data_list = env.generate(
            num_of_questions=args.num,
            max_attempts=args.max_attempts,
            difficulty=diff,
        )
        elapsed = time.time() - t0

        out_path = os.path.join(args.outdir, f"difficulty_{diff}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(item.to_json_str() + "\n")

        print(
            f"  → {len(data_list)}/{args.num} puzzles saved to {out_path}  "
            f"({elapsed:.1f}s)"
        )

    print("\nDone! ✅")


if __name__ == "__main__":
    main()
