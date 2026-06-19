"""RAG mitigation: transcribe page with vision model, chunk, hybrid-retrieve, inject context."""

from __future__ import annotations

import threading
from collections import defaultdict
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


_RETRIEVAL_MODES = {"dense", "bm25", "hybrid"}


class RagRetriever:
    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 4,
        chunk_max_chars: int = 400,
        transcribe_max_tokens: int = 1024,
        cache_dir: str = "data/ocr_cache",
        retrieval_mode: str = "hybrid",
    ) -> None:
        if retrieval_mode not in _RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {_RETRIEVAL_MODES}, got {retrieval_mode!r}")
        self._embed_model_name = embed_model
        self._top_k = top_k
        self._chunk_max_chars = chunk_max_chars
        self._transcribe_max_tokens = transcribe_max_tokens
        self._cache_dir = Path(cache_dir)
        self._retrieval_mode = retrieval_mode
        self._embedder = None  # lazy-loaded SentenceTransformer
        self._embedder_lock = threading.Lock()
        self._transcribe_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    def _cache_key(self, item: dict) -> str:
        doc = item["document_path"].replace("/", "_").replace(".", "_")
        page = item.get("page_index", 0)
        return f"{doc}_p{page}"

    def transcribe(self, item: dict, model) -> str:
        """Return page text, using disk cache to avoid repeated model calls.

        Locked per cache key so concurrent threads racing on the same uncached
        document/page don't both issue the (expensive) transcription call.
        """
        key = self._cache_key(item)
        cache_file = self._cache_dir / f"{key}.txt"
        if cache_file.exists():
            return cache_file.read_text()
        with self._transcribe_locks[key]:
            if cache_file.exists():
                return cache_file.read_text()
            text = model.generate(
                item["document_path"],
                _TRANSCRIBE_PROMPT,
                page_index=item.get("page_index", 0),
                max_tokens=self._transcribe_max_tokens,
            )
            if text.startswith("[SKIPPED:"):
                # Transient failure (e.g. request timeout) — don't cache it as if it were
                # a real transcript, so the next attempt retries instead of reusing garbage.
                return ""
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

    def _bm25_ranks(self, chunk_list: list[str], question: str) -> dict[int, int]:
        from rank_bm25 import BM25Okapi
        tokenized = [c.lower().split() for c in chunk_list]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(question.lower().split())
        order = sorted(range(len(chunk_list)), key=lambda i: scores[i], reverse=True)
        return {idx: rank for rank, idx in enumerate(order)}

    def _dense_ranks(self, chunk_list: list[str], question: str) -> dict[int, int]:
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    from sentence_transformers import SentenceTransformer
                    self._embedder = SentenceTransformer(self._embed_model_name, device="cpu")
        # encode() is not safe to call concurrently from multiple threads: the HF fast
        # tokenizer's internal Rayon thread pool can deadlock when invoked from several
        # Python threads at once, so all calls are serialized behind _embedder_lock.
        with self._embedder_lock:
            embeddings = self._embedder.encode(chunk_list, normalize_embeddings=True)
            q_emb = self._embedder.encode([question], normalize_embeddings=True)[0]
        scores = embeddings @ q_emb
        order = sorted(range(len(chunk_list)), key=lambda i: float(scores[i]), reverse=True)
        return {idx: rank for rank, idx in enumerate(order)}

    def retrieve(self, item: dict, question: str, model) -> list[str]:
        """Retrieve top-k chunks using the configured retrieval_mode (dense | bm25 | hybrid)."""
        text = self.transcribe(item, model)
        chunk_list = self.chunks(text)
        if not chunk_list:
            return []

        if self._retrieval_mode == "bm25":
            ranks = self._bm25_ranks(chunk_list, question)
            top = sorted(ranks, key=lambda i: ranks[i])
        elif self._retrieval_mode == "dense":
            ranks = self._dense_ranks(chunk_list, question)
            top = sorted(ranks, key=lambda i: ranks[i])
        else:  # hybrid — RRF fusion (k=60)
            sparse_ranks = self._bm25_ranks(chunk_list, question)
            dense_ranks = self._dense_ranks(chunk_list, question)
            k = 60
            rrf = {
                i: 1.0 / (k + sparse_ranks[i]) + 1.0 / (k + dense_ranks[i])
                for i in range(len(chunk_list))
            }
            top = sorted(rrf, key=lambda i: rrf[i], reverse=True)

        return [chunk_list[i] for i in top[: self._top_k]]


class RagStrategy(MitigationStrategy):
    name = "rag"

    def __init__(self, config: dict) -> None:
        self.retriever = RagRetriever(
            embed_model=config.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            top_k=config.get("top_k", 4),
            chunk_max_chars=config.get("chunk_max_chars", 400),
            transcribe_max_tokens=config.get("transcribe_max_tokens", 1024),
            cache_dir=config.get("cache_dir", "data/ocr_cache"),
            retrieval_mode=config.get("retrieval_mode", "hybrid"),
        )

    def build_prompt(self, item: dict, model) -> str:
        from .base import get_question
        chunks = self.retriever.retrieve(item, get_question(item), model)
        context = "\n".join(f"- {c}" for c in chunks) if chunks else "(no passages retrieved)"
        # Document text can itself contain "{"/"}" (e.g. raw JSON snippets in a scanned
        # table). Escape them so the literal braces survive the *second* .format() call
        # done downstream in vllm_model.py (which fills in {question} on the full prompt).
        context = context.replace("{", "{{").replace("}", "}}")
        return _RAG_TEMPLATE.format(context=context)
