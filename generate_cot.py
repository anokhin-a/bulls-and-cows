#!/usr/bin/env python3
"""
Generate Chain-of-Thought (CoT) reasoning traces for Bulls & Cows puzzles
using a thinking LLM (Qwen3-Next-80B-A3B-Thinking) via vLLM OpenAI API.

Pipeline:
1. Generate new difficulty-1 puzzles using BullsCowsEnv
2. Send each puzzle to the thinking model
3. The model reasons in <think>...</think> (discarded), then outputs concise CoT
4. Verify the extracted answer matches the gold answer
5. Save correct traces to JSONL for SFT training

Usage:
    python generate_cot.py \
        --target-correct 1000 \
        --num-puzzles 1500 \
        --output data/sft_cot_difficulty_1.jsonl \
        --api-base http://localhost:8000/v1 \
        --model Qwen/Qwen3-Next-80B-A3B-Thinking \
        --concurrency 16
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bulls_and_cows.bulls_and_cows import BullsCowsEnv
from bulls_and_cows.scripts.verifier import BullsCowsVerifier
from base.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert puzzle solver. Solve the Bulls and Cows puzzle step by step.

Provide a clear, concise chain of reasoning that:
1. Analyzes each clue to determine which digits are included/excluded
2. Determines the position of each digit
3. Verifies the final answer against ALL clues

End with a verification table and your final answer in \\boxed{}.

Be concise but thorough — aim for a clear deduction that another person could follow."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def strip_think_tags(response: str) -> str:
    """Remove <think>...</think> blocks from the response, keep only the answer part."""
    # Remove all <think>...</think> blocks (greedy within each block)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return cleaned.strip()


def extract_answer(response: str) -> str:
    """Extract the answer from \\boxed{...} in the response."""
    verifier = BullsCowsVerifier()
    return verifier.extract_answer(response)


def verify_answer(data: Data, model_answer: str) -> bool:
    """Check if model's answer matches gold and is consistent with clues."""
    if model_answer != data.answer:
        return False
    # Also verify consistency with clues
    clues = data.metadata.get("clues", [])
    for clue in clues:
        bulls = sum(s == g for s, g in zip(model_answer, clue["guess"]))
        common = sum(min(model_answer.count(d), clue["guess"].count(d)) for d in set(clue["guess"]))
        cows = common - bulls
        if bulls != clue["bulls"] or cows != clue["cows"]:
            return False
    return True


@dataclass
class Stats:
    """Track generation statistics."""
    total_sent: int = 0
    correct: int = 0
    incorrect: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)

    def rate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.total_sent / elapsed if elapsed > 0 else 0

    def accuracy(self) -> float:
        answered = self.correct + self.incorrect
        return self.correct / answered if answered > 0 else 0

    def __str__(self) -> str:
        return (
            f"sent={self.total_sent} correct={self.correct} "
            f"incorrect={self.incorrect} errors={self.errors} "
            f"accuracy={self.accuracy():.1%} rate={self.rate():.1f}/s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core generation
# ─────────────────────────────────────────────────────────────────────────────

async def generate_one(
    client: AsyncOpenAI,
    model: str,
    data: Data,
    max_tokens: int,
    temperature: float,
) -> tuple[bool, str, str]:
    """
    Send one puzzle to the model and return (success, cot_response, extracted_answer).
    
    Returns:
        (True, cot_text, answer) if model answered correctly
        (False, cot_text, answer) if model answered incorrectly
        (False, "", "") on error
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": data.question},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        full_response = response.choices[0].message.content or ""
        
        # Strip <think>...</think> to get the concise CoT
        cot_response = strip_think_tags(full_response)
        
        if not cot_response:
            return False, "", ""

        # Extract and verify answer
        model_answer = extract_answer(cot_response)
        if not model_answer:
            return False, cot_response, ""

        is_correct = verify_answer(data, model_answer)
        return is_correct, cot_response, model_answer

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False, "", ""


async def process_batch(
    client: AsyncOpenAI,
    model: str,
    puzzles: list[Data],
    stats: Stats,
    target_correct: int,
    results: list[dict],
    output_path: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    checkpoint_every: int,
):
    """Process puzzles with bounded concurrency, stopping when target is reached."""
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(data: Data):
        if stats.correct >= target_correct:
            return

        async with semaphore:
            if stats.correct >= target_correct:
                return

            success, cot_response, answer = await generate_one(
                client, model, data, max_tokens, temperature
            )
            stats.total_sent += 1

            if not cot_response:
                stats.errors += 1
            elif success:
                stats.correct += 1
                result = {
                    "question": data.question,
                    "answer": data.answer,
                    "difficulty": data.difficulty,
                    "metadata": data.metadata,
                    "model_response": cot_response,
                }
                results.append(result)

                # Checkpoint
                if stats.correct % checkpoint_every == 0:
                    save_results(results, output_path)
                    print(f"  [CHECKPOINT] {stats}")
            else:
                stats.incorrect += 1

            # Progress logging
            if stats.total_sent % 50 == 0:
                print(f"  [PROGRESS] {stats}")

    tasks = [asyncio.create_task(process_one(puzzle)) for puzzle in puzzles]
    await asyncio.gather(*tasks)


def save_results(results: list[dict], output_path: str):
    """Save results to JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def async_main(args):
    print(f"=== CoT Data Generation for Bulls & Cows ===")
    print(f"Model: {args.model}")
    print(f"Target correct: {args.target_correct}")
    print(f"Puzzles to generate: {args.num_puzzles}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {args.output}")
    print()

    # --- Step 1: Generate puzzles ---
    print(f"[1/3] Generating {args.num_puzzles} difficulty-1 puzzles...")
    random.seed(args.seed)
    env = BullsCowsEnv()
    puzzles_data = env.generate(
        num_of_questions=args.num_puzzles,
        max_attempts=300,
        difficulty=1,
    )
    print(f"  Generated {len(puzzles_data)} puzzles")

    if len(puzzles_data) < args.target_correct:
        print(f"  WARNING: only generated {len(puzzles_data)} puzzles, "
              f"but target is {args.target_correct} correct answers")

    # --- Step 2: Load existing results if resuming ---
    results = []
    if args.resume and os.path.exists(args.output):
        print(f"  Resuming from {args.output}...")
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        print(f"  Loaded {len(results)} existing correct results")

        # Filter out puzzles that were already solved
        solved_answers = {r["answer"] for r in results}
        # Use question text to deduplicate (since we generate new puzzles)
        solved_questions = {r["question"] for r in results}
        puzzles_data = [p for p in puzzles_data if p.question not in solved_questions]
        print(f"  {len(puzzles_data)} puzzles remaining after dedup")

    remaining_target = args.target_correct - len(results)
    if remaining_target <= 0:
        print(f"  Already have {len(results)} correct results, target met!")
        return

    # --- Step 3: Generate CoT traces ---
    print(f"\n[2/3] Generating CoT traces (target: {remaining_target} more correct)...")
    print(f"  API base: {args.api_base}")

    client = AsyncOpenAI(
        base_url=args.api_base,
        api_key=args.api_key or "dummy",  # vLLM doesn't need a real key
    )

    stats = Stats()

    await process_batch(
        client=client,
        model=args.model,
        puzzles=puzzles_data,
        stats=stats,
        target_correct=remaining_target,
        results=results,
        output_path=args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        checkpoint_every=args.checkpoint_every,
    )

    # --- Step 4: Final save ---
    save_results(results, args.output)
    print(f"\n[3/3] Final results:")
    print(f"  {stats}")
    print(f"  Total correct saved: {len(results)}")
    print(f"  Output: {args.output}")

    if len(results) < args.target_correct:
        print(f"\n  WARNING: Only collected {len(results)}/{args.target_correct} correct samples.")
        print(f"  You may need to re-run with --resume and more --num-puzzles.")
    else:
        print(f"\n  ✅ Successfully collected {len(results)} correct CoT traces!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT reasoning traces for Bulls & Cows SFT training"
    )
    parser.add_argument(
        "--target-correct", type=int, default=1000,
        help="Number of correct CoT traces to collect (default: 1000)",
    )
    parser.add_argument(
        "--num-puzzles", type=int, default=1500,
        help="Number of puzzles to generate as input (default: 1500, "
             "should be higher than target to account for model errors)",
    )
    parser.add_argument(
        "--output", type=str, default="data/sft_cot_difficulty_1.jsonl",
        help="Output JSONL path (default: data/sft_cot_difficulty_1.jsonl)",
    )
    parser.add_argument(
        "--api-base", type=str, default="http://localhost:8000/v1",
        help="vLLM OpenAI API base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key (default: dummy for local vLLM)",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-Next-80B-A3B-Thinking",
        help="Model name (default: Qwen/Qwen3-Next-80B-A3B-Thinking)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=15000,
        help="Max tokens for model response, includes <think> + answer (default: 15000)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=16,
        help="Max concurrent requests to vLLM (default: 16)",
    )
    parser.add_argument(
        "--seed", type=int, default=12345,
        help="Random seed for puzzle generation (default: 12345, "
             "different from existing datasets to get new puzzles)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=50,
        help="Save checkpoint every N correct samples (default: 50)",
    )
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
