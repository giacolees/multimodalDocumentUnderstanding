from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class QASample:
    sample_id: str
    document_path: str          # path to image or PDF page
    question: str
    answer: str
    page_index: int = 0
    metadata: dict = field(default_factory=dict)


class BaseLoader(ABC):
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    @abstractmethod
    def load(self) -> Iterator[QASample]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
