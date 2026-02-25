#!/usr/bin/env python3
"""
Build an SFT dataset from generated CoT chains for cold SFT warmup before RL.

Reads sft_cot_difficulty_1.jsonl, extracts reasoning (before </think>),
wraps the correct answer in <answer></answer>, applies a system prompt,
filters out the top 20% longest reasoning chains (by token count).

Output: JSONL with chat-formatted messages.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer


SYSTEM_PROMPT = """Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>"""


def extract_thinking(model_response: str) -> str | None:
    """Extract the reasoning content before </think> (the thinking part)."""
    marker = "</think>"
    idx = model_response.find(marker)
    if idx == -1:
        return None
    thinking = model_response[:idx].strip()
    # Remove leading <think> tag if present (some responses start with it, some don't)
    if thinking.startswith("<think>"):
        thinking = thinking[len("<think>"):].strip()
    return thinking


def build_assistant_response(thinking: str, answer: str) -> str:
    """Build the full assistant response in the required format."""
    return f"<think>\n{thinking}\n</think>\n<answer>\n{answer}\n</answer>"


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT CoT dataset with reasoning chains"
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
        default="data/sft_cot_dataset_difficulty_1.jsonl",
        help="Path to the output SFT dataset JSONL file",
    )
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Tokenizer to use for measuring token lengths",
    )
    parser.add_argument(
        "--filter-percentile",
        type=float,
        default=80.0,
        help="Keep only chains at or below this percentile by token length (default: 80 = drop top 20%%)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Read input
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

    # Build SFT examples and compute token lengths
    sft_examples = []
    skipped = 0
    for entry in entries:
        model_response = entry.get("model_response", "")
        if not model_response:
            skipped += 1
            continue

        thinking = extract_thinking(model_response)
        if thinking is None or not thinking:
            skipped += 1
            continue

        answer = entry["answer"]
        assistant_content = build_assistant_response(thinking, answer)

        # Tokenize just the reasoning part to measure length
        thinking_tokens = tokenizer.encode(thinking, add_special_tokens=False)
        thinking_token_count = len(thinking_tokens)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": entry["question"]},
            {"role": "assistant", "content": assistant_content},
        ]

        sft_examples.append({
            "messages": messages,
            "answer": answer,
            "difficulty": entry.get("difficulty"),
            "thinking_token_count": thinking_token_count,
        })

    print(f"  Built {len(sft_examples)} SFT examples (skipped {skipped})")

    # Analyze token lengths
    token_counts = np.array([ex["thinking_token_count"] for ex in sft_examples])
    print(f"\n--- Reasoning Chain Token Length Stats (before filtering) ---")
    print(f"  Count:  {len(token_counts)}")
    print(f"  Mean:   {token_counts.mean():.0f}")
    print(f"  Median: {np.median(token_counts):.0f}")
    print(f"  Std:    {token_counts.std():.0f}")
    print(f"  Min:    {token_counts.min()}")
    print(f"  Max:    {token_counts.max()}")
    for p in [25, 50, 75, 80, 90, 95]:
        print(f"  P{p}:    {np.percentile(token_counts, p):.0f}")

    # Filter: drop top 20% longest chains
    cutoff = np.percentile(token_counts, args.filter_percentile)
    print(f"\n--- Filtering ---")
    print(f"  Cutoff (P{args.filter_percentile:.0f}): {cutoff:.0f} tokens")

    filtered_examples = [
        ex for ex in sft_examples
        if ex["thinking_token_count"] <= cutoff
    ]
    print(f"  Before filter: {len(sft_examples)}")
    print(f"  After filter:  {len(filtered_examples)}")
    print(f"  Dropped:       {len(sft_examples) - len(filtered_examples)}")

    # Stats after filtering
    filtered_counts = np.array([ex["thinking_token_count"] for ex in filtered_examples])
    print(f"\n--- Token Length Stats (after filtering) ---")
    print(f"  Mean:   {filtered_counts.mean():.0f}")
    print(f"  Median: {np.median(filtered_counts):.0f}")
    print(f"  Max:    {filtered_counts.max()}")

    # Remove internal field before saving
    for ex in filtered_examples:
        del ex["thinking_token_count"]

    # Show a sample
    if filtered_examples:
        sample = filtered_examples[0]
        assistant_preview = sample["messages"][-1]["content"][:300]
        print(f"\n  Sample assistant response (first 300 chars):")
        print(f"  {assistant_preview}...")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in filtered_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\nSaved SFT dataset to: {output_path}")
    print(f"  Total examples: {len(filtered_examples)}")


if __name__ == "__main__":
    main()
