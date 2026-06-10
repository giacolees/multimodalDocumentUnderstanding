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

    def _doc_path(self, doc_id: str, page: int) -> str:
        """Return the best available path for a document page."""
        # PDF layout (full dataset)
        pdf_path = self.data_dir / "pdfs" / doc_id
        if pdf_path.exists():
            return str(pdf_path)
        # Image layout (public subset — one PNG per page)
        img_path = self.data_dir / "documents" / f"{doc_id}_page{page}.png"
        if img_path.exists():
            return str(img_path)
        # Fall back to page 0 if the specific page image is missing
        img_page0 = self.data_dir / "documents" / f"{doc_id}_page0.png"
        if img_page0.exists():
            return str(img_page0)
        return str(pdf_path)  # let the pipeline surface a clear missing-file error

    def load(self) -> Iterator[QASample]:
        with open(self._annotations_path) as f:
            data = json.load(f)
        for item in data:
            for qa in item["qa_pairs"]:
                page = qa.get("page", 0)
                yield QASample(
                    sample_id=qa["questionId"],
                    document_path=self._doc_path(item["docId"], page),
                    question=qa["question"],
                    answer=qa["answers"][0] if qa.get("answers") else "",
                    page_index=page,
                    metadata={"docId": item["docId"], "num_pages": item.get("num_pages", 1)},
                )

    def __len__(self) -> int:
        with open(self._annotations_path) as f:
            data = json.load(f)
        return sum(len(item["qa_pairs"]) for item in data)
