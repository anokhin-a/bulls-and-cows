import random
from itertools import permutations
from typing import Optional

from base.env import Env
from base.data import Data
from bulls_and_cows.scripts.verifier import BullsCowsVerifier
from bulls_and_cows.scripts.prompt import prompt_bulls_and_cows


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty → hyperparameter mapping
# ─────────────────────────────────────────────────────────────────────────────

DIFFICULTY_MAP: dict[int, dict] = {
    1:  {"num_digits": 3, "num_clues": 5},
    2:  {"num_digits": 3, "num_clues": 4},
    3:  {"num_digits": 4, "num_clues": 6},
    4:  {"num_digits": 4, "num_clues": 5},
    5:  {"num_digits": 4, "num_clues": 4},
    6:  {"num_digits": 5, "num_clues": 6},
    7:  {"num_digits": 5, "num_clues": 5},
    8:  {"num_digits": 5, "num_clues": 4},
    9:  {"num_digits": 6, "num_clues": 5},
    10: {"num_digits": 6, "num_clues": 4},
}


class BullsCowsEnv(Env):
    """
    Bulls-and-Cows environment for LLM agents.

    A secret number of N distinct digits (from 0-9) is chosen.
    K clues (previous guesses with bull/cow scores) are provided.
    The LLM must determine the secret number in a single step.
    The generator guarantees that the solution is unique.
    """

    def __init__(self):
        super().__init__(name="bulls_and_cows", verifier=BullsCowsVerifier)

    # ── core helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _count_bulls_cows(secret: str, guess: str) -> tuple[int, int]:
        """Return (bulls, cows) for a guess against a secret."""
        bulls = sum(s == g for s, g in zip(secret, guess))
        common = sum(min(secret.count(d), guess.count(d)) for d in set(guess))
        cows = common - bulls
        return bulls, cows

    @staticmethod
    def _random_number(num_digits: int) -> str:
        """Generate a random number with `num_digits` distinct digits."""
        digits = random.sample(range(10), num_digits)
        return "".join(map(str, digits))

    @staticmethod
    def _all_candidates(num_digits: int) -> list[str]:
        """Return all possible N-digit numbers with distinct digits."""
        return [
            "".join(map(str, p))
            for p in permutations(range(10), num_digits)
        ]

    def _count_consistent(self, clues: list[dict], num_digits: int) -> list[str]:
        """
        Return all candidate numbers consistent with every clue.
        """
        candidates = self._all_candidates(num_digits)
        consistent = []
        for candidate in candidates:
            ok = True
            for clue in clues:
                b, c = self._count_bulls_cows(candidate, clue["guess"])
                if b != clue["bulls"] or c != clue["cows"]:
                    ok = False
                    break
            if ok:
                consistent.append(candidate)
        return consistent

    def _generate_one(self, num_digits: int, num_clues: int,
                      max_attempts: int) -> Optional[Data]:
        """
        Try to generate a single puzzle with a unique solution.

        @return: Data if successful, None if max_attempts exceeded.
        """
        difficulty_val = None
        # Determine difficulty for metadata from the map
        for d, params in DIFFICULTY_MAP.items():
            if params["num_digits"] == num_digits and params["num_clues"] == num_clues:
                difficulty_val = d
                break
        if difficulty_val is None:
            difficulty_val = 1  # default if custom hyperparams

        for _ in range(max_attempts):
            secret = self._random_number(num_digits)

            # Generate random distinct guesses
            clues = []
            seen_guesses = {secret}  # avoid guessing the secret itself
            attempts = 0
            while len(clues) < num_clues and attempts < num_clues * 10:
                guess = self._random_number(num_digits)
                attempts += 1
                if guess in seen_guesses:
                    continue
                seen_guesses.add(guess)
                bulls, cows = self._count_bulls_cows(secret, guess)
                clues.append({
                    "guess": guess,
                    "bulls": bulls,
                    "cows": cows,
                })

            if len(clues) < num_clues:
                continue  # could not generate enough distinct guesses

            # Check uniqueness
            consistent = self._count_consistent(clues, num_digits)
            if len(consistent) == 1 and consistent[0] == secret:
                question = prompt_bulls_and_cows(num_digits, clues)
                return Data(
                    question=question,
                    answer=secret,
                    difficulty=difficulty_val,
                    metadata={
                        "num_digits": num_digits,
                        "num_clues": num_clues,
                        "clues": clues,
                        "secret": secret,
                    },
                )

        return None  # failed after max_attempts

    # ── public interface ──────────────────────────────────────────────────

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = None,
        **kwargs,
    ) -> list[Data]:
        """
        Generate game questions and answers.

        Supports two ways to control difficulty:
        1. Pass `difficulty` (int 1-10) — automatically maps to num_digits / num_clues.
        2. Pass hyperparameters directly via kwargs:
           `num_digits` (int) and `num_clues` (int).
           These override the difficulty mapping.

        @param num_of_questions: number of puzzles to generate
        @param max_attempts: attempts per puzzle before giving up
        @param difficulty: difficulty level 1-10 (optional)
        @param kwargs: direct hyperparameters (num_digits, num_clues)
        @return: list of Data
        """
        # Resolve hyperparameters
        if "num_digits" in kwargs and "num_clues" in kwargs:
            num_digits = kwargs["num_digits"]
            num_clues = kwargs["num_clues"]
        elif difficulty is not None:
            if difficulty not in DIFFICULTY_MAP:
                raise ValueError(
                    f"difficulty must be in [1, 10], got {difficulty}"
                )
            params = DIFFICULTY_MAP[difficulty]
            num_digits = params["num_digits"]
            num_clues = params["num_clues"]
        else:
            # Default to difficulty 1
            params = DIFFICULTY_MAP[1]
            num_digits = params["num_digits"]
            num_clues = params["num_clues"]

        results: list[Data] = []
        for i in range(num_of_questions):
            data = self._generate_one(num_digits, num_clues, max_attempts)
            if data is not None:
                results.append(data)

        return results

    def extract_answer(self, test_solution: str) -> str:
        """
        Extract the answer from the test solution (delegates to verifier).
        """
        return self.verifier.extract_answer(test_solution)
