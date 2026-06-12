"""RAG mitigation: transcribe page with vision model, chunk, hybrid-retrieve, inject context."""

from __future__ import annotations

from pathlib import Path

from .base import MitigationStrategy

_TRANSCRIBE_PROMPT = (
    "Transcribe all visible text from this document page faithfully. "
    "Preserve table labels, column headers, numbers, and layout markers exactly as shown. "
    "Output plain text only, no commentary."
)

_RAG_TEMPLATE = (
    "Here are the passages retrieved from this document most relevant to the question:\n"
    "{context}\n\n"
    "You are also shown the full document image. "
    "If the answer is not supported by the document, respond UNANSWERABLE; "
    "otherwise provide the answer.\n\n"
    "Question: {{question}}"
)


class RagRetriever:
    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 4,
        chunk_max_chars: int = 400,
        transcribe_max_tokens: int = 1024,
        cache_dir: str = "data/ocr_cache",
    ) -> None:
        self._embed_model_name = embed_model
        self._top_k = top_k
        self._chunk_max_chars = chunk_max_chars
        self._transcribe_max_tokens = transcribe_max_tokens
        self._cache_dir = Path(cache_dir)
        self._embedder = None  # lazy-loaded SentenceTransformer

    def _cache_key(self, item: dict) -> str:
        doc = item["document_path"].replace("/", "_").replace(".", "_")
        page = item.get("page_index", 0)
        return f"{doc}_p{page}"

    def transcribe(self, item: dict, model) -> str:
        """Return page text, using disk cache to avoid repeated model calls."""
        key = self._cache_key(item)
        cache_file = self._cache_dir / f"{key}.txt"
        if cache_file.exists():
            return cache_file.read_text()
        text = model.generate(
            item["document_path"],
            _TRANSCRIBE_PROMPT,
            page_index=item.get("page_index", 0),
            max_tokens=self._transcribe_max_tokens,
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text)
        return text

    def chunks(self, text: str) -> list[str]:
        """Split text into chunks of at most chunk_max_chars, preserving as much content as possible."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        lines: list[str] = []
        for p in paragraphs:
            if len(p) <= self._chunk_max_chars:
                lines.append(p)
            else:
                for line in p.split("\n"):
                    if line.strip():
                        lines.append(line.strip())
        result: list[str] = []
        current = ""
        for line in lines:
            candidate = (current + "\n" + line).strip() if current else line
            if len(candidate) <= self._chunk_max_chars:
                current = candidate
            else:
                if current:
                    result.append(current)
                current = line[: self._chunk_max_chars]
        if current:
            result.append(current)
        if not result:
            stripped = text.strip()
            return [stripped[: self._chunk_max_chars]] if stripped else []
        return result

    def retrieve(self, item: dict, question: str, model) -> list[str]:
        """Stub — implemented in Task 7."""
        raise NotImplementedError
