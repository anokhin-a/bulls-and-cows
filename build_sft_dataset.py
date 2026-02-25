#!/usr/bin/env python3
"""
Build an SFT dataset from generated CoT chains.

Reads the CoT JSONL file (e.g. sft_cot_difficulty_1.jsonl), extracts the
response part AFTER </think> from the model_response field, and creates
a chat-formatted SFT dataset suitable for fine-tuning.

Output format: JSONL where each line is a JSON object with a "messages" list
containing system, user, and assistant messages.
"""

import json
import re
import argparse
from pathlib import Path


SYSTEM_PROMPT = """\
You are a skilled puzzle solver. Analyze the Bulls and Cows clues carefully, \
use logical deduction to determine the secret number, and present your \
reasoning clearly before giving the final answer."""


def extract_after_think(model_response: str) -> str:
    """Extract everything after the </think> tag from model_response."""
    marker = "</think>"
    idx = model_response.find(marker)
    if idx == -1:
        # No </think> tag — return the full response as-is
        return model_response.strip()
    return model_response[idx + len(marker):].strip()


def build_sft_example(entry: dict) -> dict:
    """
    Build a single SFT training example in chat format.

    Returns a dict with:
      - "messages": [{role, content}, ...]
      - "question", "answer" — kept for reference / filtering
    """
    question = entry["question"]
    answer = entry["answer"]
    model_response = entry.get("model_response", "")

    assistant_content = extract_after_think(model_response)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        "messages": messages,
        "answer": answer,
        "difficulty": entry.get("difficulty"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT dataset from CoT chains"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/sft_cot_difficulty_1.jsonl",
        help="Path to the input CoT JSONL file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/sft_dataset_difficulty_1.jsonl",
        help="Path to the output SFT dataset JSONL file",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Omit system prompt from the messages",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    print(f"Reading from: {input_path}")

    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping line {line_num} (JSON error: {e})")

    print(f"  Loaded {len(entries)} entries")

    # Build SFT examples
    sft_examples = []
    skipped = 0
    for entry in entries:
        model_response = entry.get("model_response", "")
        if not model_response:
            skipped += 1
            continue

        sft_example = build_sft_example(entry)

        # Remove system prompt if requested
        if args.no_system_prompt:
            sft_example["messages"] = [
                m for m in sft_example["messages"] if m["role"] != "system"
            ]

        # Sanity check: assistant response should not be empty
        assistant_content = sft_example["messages"][-1]["content"]
        if not assistant_content:
            skipped += 1
            continue

        sft_examples.append(sft_example)

    print(f"  Built {len(sft_examples)} SFT examples (skipped {skipped})")

    # Show a sample
    if sft_examples:
        sample = sft_examples[0]
        assistant_preview = sample["messages"][-1]["content"][:200]
        print(f"\n  Sample assistant response (first 200 chars):")
        print(f"  {assistant_preview}...")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in sft_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\nSaved SFT dataset to: {output_path}")
    print(f"  Total examples: {len(sft_examples)}")

    # Stats on response lengths
    lengths = [
        len(ex["messages"][-1]["content"]) for ex in sft_examples
    ]
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        print(f"  Avg assistant response length: {avg_len:.0f} chars")
        print(f"  Min: {min(lengths)}, Max: {max(lengths)}")


if __name__ == "__main__":
    main()
