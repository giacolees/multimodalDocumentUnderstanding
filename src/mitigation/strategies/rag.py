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
        """Split text into chunks of at most chunk_max_chars, preserving paragraph boundaries."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        result: list[str] = []
        for p in paragraphs:
            if len(p) <= self._chunk_max_chars:
                result.append(p)
            else:
                # Oversized paragraph: split by lines and greedily merge within max_chars
                lines = [line.strip() for line in p.split("\n") if line.strip()]
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
        """Hybrid BM25 + dense retrieval with RRF fusion."""
        text = self.transcribe(item, model)
        chunk_list = self.chunks(text)
        if not chunk_list:
            return []

        # Sparse ranking (BM25)
        from rank_bm25 import BM25Okapi
        tokenized = [c.lower().split() for c in chunk_list]
        bm25 = BM25Okapi(tokenized)
        sparse_scores = bm25.get_scores(question.lower().split())
        sparse_order = sorted(range(len(chunk_list)),
                              key=lambda i: sparse_scores[i], reverse=True)
        sparse_ranks: dict[int, int] = {idx: rank for rank, idx in enumerate(sparse_order)}

        # Dense ranking (SentenceTransformer)
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embed_model_name)
        import numpy as np
        embeddings = self._embedder.encode(chunk_list, normalize_embeddings=True)
        q_emb = self._embedder.encode([question], normalize_embeddings=True)[0]
        dense_scores_arr = embeddings @ q_emb
        dense_order = sorted(range(len(chunk_list)),
                             key=lambda i: float(dense_scores_arr[i]), reverse=True)
        dense_ranks: dict[int, int] = {idx: rank for rank, idx in enumerate(dense_order)}

        # RRF fusion (k=60, parameter-free)
        k = 60
        rrf = {
            i: 1.0 / (k + sparse_ranks[i]) + 1.0 / (k + dense_ranks[i])
            for i in range(len(chunk_list))
        }
        top = sorted(rrf, key=lambda i: rrf[i], reverse=True)
        return [chunk_list[i] for i in top[: self._top_k]]
