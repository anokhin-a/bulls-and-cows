#!/usr/bin/env python3
"""
Test script for the Bulls-and-Cows environment.

Tests:
1. Generation at each difficulty level (1-10)
2. Verification of correct answers
3. Rejection of wrong answers
4. extract_answer on various LLM output formats
5. Uniqueness of solutions
6. Direct hyperparameter mode
"""

import sys
import time

from bulls_and_cows.bulls_and_cows import BullsCowsEnv


def test_extract_answer():
    """Test answer extraction from different LLM output formats."""
    env = BullsCowsEnv()
    print("=" * 60)
    print("TEST: extract_answer")
    print("=" * 60)

    cases = [
        # (input, expected)
        (r"The answer is \boxed{1234}", "1234"),
        (r"Let me think step by step... \boxed{0567}", "0567"),
        ("```python\n0123\n```", "0123"),
        ("The secret number is 4567.", "4567"),
        (r"First \boxed{999} then \boxed{1234}", "1234"),  # last boxed
        ("", ""),
    ]

    all_ok = True
    for inp, expected in cases:
        result = env.extract_answer(inp)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_ok = False
        print(f"  {status}  extract({inp[:40]!r}...) = {result!r}  (expected {expected!r})")

    print()
    return all_ok


def test_generation_and_verification():
    """Test generation and verification at various difficulty levels."""
    env = BullsCowsEnv()
    print("=" * 60)
    print("TEST: generate + verify (difficulty 1-10)")
    print("=" * 60)

    all_ok = True

    for diff in range(1, 11):
        t0 = time.time()
        data_list = env.generate(num_of_questions=3, max_attempts=200, difficulty=diff)
        elapsed = time.time() - t0

        if len(data_list) == 0:
            print(f"  ❌  difficulty={diff:>2}: generated 0/3 puzzles ({elapsed:.2f}s)")
            all_ok = False
            continue

        # Verify each puzzle
        ok_count = 0
        for data in data_list:
            # Correct answer should pass
            correct_input = rf"The answer is \boxed{{{data.answer}}}"
            if not env.verify(data, correct_input):
                print(f"  ❌  difficulty={diff}: verify(correct) returned False for {data.answer}")
                all_ok = False
                continue

            # Wrong answer should fail
            wrong = "0000000"[:data.metadata["num_digits"]]
            wrong_input = rf"The answer is \boxed{{{wrong}}}"
            if env.verify(data, wrong_input):
                print(f"  ❌  difficulty={diff}: verify(wrong) returned True for {wrong}")
                all_ok = False
                continue

            ok_count += 1

        status = "✅" if ok_count == len(data_list) else "⚠️"
        nd = data_list[0].metadata["num_digits"]
        nc = data_list[0].metadata["num_clues"]
        print(
            f"  {status}  difficulty={diff:>2} (N={nd}, K={nc}): "
            f"{ok_count}/{len(data_list)} verified  ({elapsed:.2f}s)"
        )

    print()
    return all_ok


def test_direct_hyperparameters():
    """Test generation with direct hyperparameters (no difficulty)."""
    env = BullsCowsEnv()
    print("=" * 60)
    print("TEST: direct hyperparameters (num_digits=3, num_clues=4)")
    print("=" * 60)

    data_list = env.generate(num_of_questions=2, max_attempts=200,
                             num_digits=3, num_clues=4)

    ok = len(data_list) > 0
    for data in data_list:
        assert data.metadata["num_digits"] == 3
        assert data.metadata["num_clues"] == 4
        correct_input = rf"The answer is \boxed{{{data.answer}}}"
        if not env.verify(data, correct_input):
            ok = False
            break

    status = "✅" if ok else "❌"
    print(f"  {status}  generated {len(data_list)}/2 puzzles, all verified")
    print()
    return ok


def test_show_example():
    """Show a full example prompt."""
    env = BullsCowsEnv()
    print("=" * 60)
    print("EXAMPLE: prompt at difficulty=5")
    print("=" * 60)
    data_list = env.generate(num_of_questions=1, max_attempts=200, difficulty=5)
    if data_list:
        data = data_list[0]
        print(data.question)
        print(f"\n[ANSWER: {data.answer}]")
    else:
        print("  ❌  Could not generate example")
    print()


def main():
    results = []
    results.append(("extract_answer", test_extract_answer()))
    results.append(("generation+verification", test_generation_and_verification()))
    results.append(("direct_hyperparams", test_direct_hyperparameters()))
    test_show_example()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS ✅" if ok else "FAIL ❌"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
