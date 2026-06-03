from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import random


class CorruptionType(str, Enum):
    NLP_ENTITY = "nlp_entity"
    ELEMENT = "element"
    LAYOUT = "layout"


@dataclass
class CorruptedSample:
    original_question: str
    corrupted_question: str
    corruption_type: CorruptionType
    corruption_detail: str          # e.g. "year→place", "Table 1→Figure 2"
    is_unanswerable: bool = True


class BaseCorruptor(ABC):
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    @abstractmethod
    def corrupt(self, question: str, context: Optional[dict] = None) -> Optional[CorruptedSample]:
        """Return a corrupted sample, or None if this corruptor cannot handle the question."""
        ...

    @property
    @abstractmethod
    def corruption_type(self) -> CorruptionType:
        ...
