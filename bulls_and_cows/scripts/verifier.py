import re

from base.data import Data
from base.verifier import Verifier


class BullsCowsVerifier(Verifier):
    """
    Verifier for the Bulls-and-Cows game.
    """

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _count_bulls_cows(secret: str, guess: str) -> tuple[int, int]:
        """Return (bulls, cows) for a guess against a secret."""
        bulls = sum(s == g for s, g in zip(secret, guess))
        # cows = digits present in both but not in the same position
        common = sum(min(secret.count(d), guess.count(d)) for d in set(guess))
        cows = common - bulls
        return bulls, cows

    # ── interface ─────────────────────────────────────────────────────────

    def extract_answer(self, test_solution: str) -> str:
        """
        Extract the answer from an LLM response.

        Strategy (in priority order):
        1. Look for \\boxed{...} — take the LAST occurrence
        2. Look for a ```python ... ``` code block
        3. Fallback: find the last sequence of digits that could be an answer
        """
        if not test_solution:
            return ""

        # 1. \boxed{...}  (last match)
        boxed_pattern = r"\\boxed\{([^}]+)\}"
        matches = re.findall(boxed_pattern, test_solution)
        if matches:
            return matches[-1].strip()

        # 2. python code block (last match)
        code_pattern = r"```python\s*([\s\S]*?)\s*```"
        matches = re.findall(code_pattern, test_solution)
        if matches:
            # Extract only digits from the code block
            digits = re.findall(r"\d+", matches[-1])
            if digits:
                return digits[-1].strip()

        # 3. Fallback: last standalone digit sequence of length >= 3
        digit_sequences = re.findall(r"\b\d{3,}\b", test_solution)
        if digit_sequences:
            return digit_sequences[-1].strip()

        return ""

    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify whether the extracted answer matches the gold answer.

        Beyond simple string comparison, we also verify that the answer
        is consistent with all clues stored in data.metadata.
        """
        try:
            test_answer = self.extract_answer(test_solution)
            if not test_answer:
                return False

            gold_answer = data.answer

            # Basic check: same string
            if test_answer != gold_answer:
                return False

            # Structural checks
            if len(set(test_answer)) != len(test_answer):
                return False  # digits must be unique

            if not test_answer.isdigit():
                return False

            # Deep check: consistent with every clue
            clues = data.metadata.get("clues", [])
            for clue in clues:
                expected_bulls = clue["bulls"]
                expected_cows = clue["cows"]
                bulls, cows = self._count_bulls_cows(test_answer, clue["guess"])
                if bulls != expected_bulls or cows != expected_cows:
                    return False

            return True

        except Exception:
            return False
