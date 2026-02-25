#!/usr/bin/env python3
"""
Validate generated CoT data: check answer correctness, response quality, stats.

Usage:
    python validate_cot.py data/sft_cot_difficulty_1.jsonl
"""

import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bulls_and_cows.scripts.verifier import BullsCowsVerifier


def count_bulls_cows(secret: str, guess: str) -> tuple[int, int]:
    bulls = sum(s == g for s, g in zip(secret, guess))
    common = sum(min(secret.count(d), guess.count(d)) for d in set(guess))
    cows = common - bulls
    return bulls, cows


def validate_file(path: str):
    print(f"Validating: {path}")
    print("=" * 60)

    verifier = BullsCowsVerifier()
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Total entries: {len(items)}")

    if not items:
        print("No data to validate.")
        return

    # Stats
    correct = 0
    incorrect = 0
    missing_answer = 0
    response_lengths = []
    has_boxed = 0
    has_verification = 0
    has_think_tags = 0

    for i, item in enumerate(items):
        response = item.get("model_response", "")
        gold_answer = item["answer"]
        clues = item["metadata"]["clues"]

        # Check for leftover think tags
        if "<think>" in response:
            has_think_tags += 1

        # Extract answer
        extracted = verifier.extract_answer(response)

        if not extracted:
            missing_answer += 1
            continue

        # Check boxed
        if "\\boxed{" in response:
            has_boxed += 1

        # Check for verification table
        if "✅" in response or "verification" in response.lower():
            has_verification += 1

        # Check correctness
        if extracted == gold_answer:
            # Verify against clues
            all_ok = True
            for clue in clues:
                b, c = count_bulls_cows(extracted, clue["guess"])
                if b != clue["bulls"] or c != clue["cows"]:
                    all_ok = False
                    break
            if all_ok:
                correct += 1
            else:
                incorrect += 1
                print(f"  [WARN] Entry {i}: answer matches but clue check fails: {extracted}")
        else:
            incorrect += 1
            print(f"  [WARN] Entry {i}: expected={gold_answer}, got={extracted}")

        response_lengths.append(len(response))

    print(f"\nResults:")
    print(f"  Correct answers:     {correct}/{len(items)} ({correct/len(items)*100:.1f}%)")
    print(f"  Incorrect answers:   {incorrect}")
    print(f"  Missing answers:     {missing_answer}")
    print(f"  Has \\boxed{{}}:      {has_boxed}")
    print(f"  Has verification:    {has_verification}")
    print(f"  Has leftover <think>: {has_think_tags}")

    if response_lengths:
        avg_len = sum(response_lengths) / len(response_lengths)
        min_len = min(response_lengths)
        max_len = max(response_lengths)
        print(f"\nResponse length (chars):")
        print(f"  Average: {avg_len:.0f}")
        print(f"  Min:     {min_len}")
        print(f"  Max:     {max_len}")

    # Show sample
    print(f"\n{'=' * 60}")
    print("Sample entry (first correct):")
    print("=" * 60)
    for item in items:
        response = item.get("model_response", "")
        extracted = verifier.extract_answer(response)
        if extracted == item["answer"]:
            print(f"Question: {item['question'][:200]}...")
            print(f"Gold answer: {item['answer']}")
            print(f"Response preview ({len(response)} chars):")
            print(response[:1500])
            if len(response) > 1500:
                print(f"... ({len(response) - 1500} more chars)")
            break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_cot.py <path_to_jsonl>")
        sys.exit(1)
    validate_file(sys.argv[1])
