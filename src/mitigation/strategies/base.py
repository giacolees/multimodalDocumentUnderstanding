from __future__ import annotations
from abc import ABC, abstractmethod


class MitigationStrategy(ABC):
    name: str

    def prepare(self, dataset: list[dict], model) -> None:
        """Optional one-time setup called once before the eval loop. No-op by default."""

    @abstractmethod
    def build_prompt(self, item: dict, model) -> str:
        """Return the prompt string for this item. Leave {question} as a literal placeholder."""
