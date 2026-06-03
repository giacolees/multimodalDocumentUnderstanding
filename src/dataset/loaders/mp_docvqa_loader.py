import json
from pathlib import Path
from typing import Iterator
from .base_loader import BaseLoader, QASample


class MPDocVQALoader(BaseLoader):
    """Loads MultiPageDocVQA dataset with sliding-window support."""

    def __init__(self, data_dir: str | Path, split: str = "val", window_size: int = 1):
        super().__init__(data_dir)
        self.split = split
        self.window_size = window_size
        self._annotations_path = self.data_dir / f"{split}_v1.0.json"

    def load(self) -> Iterator[QASample]:
        with open(self._annotations_path) as f:
            data = json.load(f)
        for item in data["data"]:
            answer_page = item.get("answer_page_idx", 0)
            # yield samples for each window that covers the answer page
            num_pages = len(item["page_ids"])
            for start in range(0, num_pages, max(1, self.window_size)):
                end = min(start + self.window_size, num_pages)
                window_pages = item["page_ids"][start:end]
                yield QASample(
                    sample_id=f"{item['questionId']}_w{start}",
                    document_path=str(self.data_dir / "images"),
                    question=item["question"],
                    answer=item["answers"][0] if item.get("answers") else "",
                    page_index=answer_page,
                    metadata={
                        "window_pages": window_pages,
                        "window_start": start,
                        "window_end": end,
                        "docId": item.get("docId", ""),
                    },
                )

    def __len__(self) -> int:
        with open(self._annotations_path) as f:
            data = json.load(f)
        return len(data["data"])
