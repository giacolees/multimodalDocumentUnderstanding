from __future__ import annotations
from abc import ABC, abstractmethod


def get_question(item: dict) -> str:
    """Return the question string regardless of dataset schema.

    Final dataset uses 'question'; corrupted-only dataset uses 'corrupted_question'.
    """
    return item.get("question") or item.get("corrupted_question", "")


class MitigationStrategy(ABC):
    name: str

    def prepare(self, dataset: list[dict], model) -> None:
        """Optional one-time setup called once before the eval loop. No-op by default."""

    @abstractmethod
    def build_prompt(self, item: dict, model) -> str:
        """Return the prompt string for this item. Leave {question} as a literal placeholder."""
