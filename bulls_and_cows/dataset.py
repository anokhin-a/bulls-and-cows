"""
PyTorch Dataset for pre-generated Bulls-and-Cows puzzles.

Each dataset stores puzzles of a single difficulty level loaded
from a JSONL file. Puzzles are fixed (not re-sampled) for reproducibility.
"""

import json
import os
from typing import Optional

from torch.utils.data import Dataset

from base.data import Data


class BullsCowsDataset(Dataset):
    """
    A PyTorch Dataset wrapping a JSONL file of Bulls-and-Cows puzzles.

    Each item is a dict with keys:
        - "question" (str): the prompt with game rules and clues
        - "answer"   (str): the secret number
        - "difficulty" (int): difficulty level
        - "metadata" (dict): clues, num_digits, num_clues, secret

    Usage:
        ds = BullsCowsDataset("data/difficulty_5.jsonl")
        item = ds[0]  # -> {"question": ..., "answer": ..., ...}
    """

    def __init__(self, jsonl_path: str):
        """
        @param jsonl_path: path to a JSONL file with pre-generated puzzles
        """
        self.jsonl_path = jsonl_path
        self.data: list[Data] = []
        self._load(jsonl_path)

    def _load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                item = Data(
                    question=obj["question"],
                    answer=obj["answer"],
                    difficulty=obj.get("difficulty", 1),
                    metadata=obj.get("metadata"),
                )
                if "gpt_response" in obj:
                    item.gpt_response = obj["gpt_response"]
                self.data.append(item)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        return {
            "question": item.question,
            "answer": item.answer,
            "difficulty": item.difficulty,
            "metadata": item.metadata,
        }

    def get_data(self, idx: int) -> Data:
        """Return the raw Data object (useful for verification)."""
        return self.data[idx]

    @property
    def difficulty(self) -> Optional[int]:
        """Return the difficulty if all items share the same one."""
        if not self.data:
            return None
        d = self.data[0].difficulty
        return d if all(item.difficulty == d for item in self.data) else None

    def __repr__(self) -> str:
        return (
            f"BullsCowsDataset(path={self.jsonl_path!r}, "
            f"size={len(self)}, difficulty={self.difficulty})"
        )
