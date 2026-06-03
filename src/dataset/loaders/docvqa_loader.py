import json
from pathlib import Path
from typing import Iterator
from .base_loader import BaseLoader, QASample


class DocVQALoader(BaseLoader):
    """Loads DocVQA dataset splits from the standard directory layout."""

    def __init__(self, data_dir: str | Path, split: str = "val"):
        super().__init__(data_dir)
        self.split = split
        self._annotations_path = self.data_dir / split / f"{split}_v1.0.json"

    def load(self) -> Iterator[QASample]:
        with open(self._annotations_path) as f:
            data = json.load(f)
        for item in data["data"]:
            for qa in item.get("qa_pairs", [item]):
                yield QASample(
                    sample_id=str(qa["questionId"]),
                    document_path=str(self.data_dir / self.split / "documents" / item["image"]),
                    question=qa["question"],
                    answer=qa["answers"][0] if qa.get("answers") else "",
                    page_index=0,
                    metadata={"docId": item.get("docId", "")},
                )

    def __len__(self) -> int:
        with open(self._annotations_path) as f:
            data = json.load(f)
        return sum(len(item.get("qa_pairs", [item])) for item in data["data"])
