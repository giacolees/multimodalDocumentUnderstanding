from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PredictionResult:
    sample_id: str
    predicted_unanswerable: bool    # True = model says unanswerable
    confidence: float               # 0.0–1.0 if available, else -1
    raw_response: str


class BaseVisionModel(ABC):
    """Common interface for all Vision LLM / Vision Transformer backends."""

    @abstractmethod
    def predict_unanswerable(
        self,
        document_path: str,
        question: str,
        prompt_template: str,
        page_indices: list[int] | None = None,
    ) -> PredictionResult:
        ...

    @abstractmethod
    def name(self) -> str:
        ...
