import json
from pathlib import Path
from typing import Iterator
from .base_loader import BaseLoader, QASample


class DUDELoader(BaseLoader):
    """Loads DUDE dataset (single- and multi-page documents)."""

    def __init__(self, data_dir: str | Path, split: str = "val"):
        super().__init__(data_dir)
        self.split = split
        self._annotations_path = self.data_dir / f"dude_{split}.json"

    def load(self) -> Iterator[QASample]:
        with open(self._annotations_path) as f:
            data = json.load(f)
        for item in data:
            for qa in item["qa_pairs"]:
                yield QASample(
                    sample_id=qa["questionId"],
                    document_path=str(self.data_dir / "pdfs" / item["docId"]),
                    question=qa["question"],
                    answer=qa["answers"][0] if qa.get("answers") else "",
                    page_index=qa.get("page", 0),
                    metadata={"docId": item["docId"], "num_pages": item.get("num_pages", 1)},
                )

    def __len__(self) -> int:
        with open(self._annotations_path) as f:
            data = json.load(f)
        return sum(len(item["qa_pairs"]) for item in data)
