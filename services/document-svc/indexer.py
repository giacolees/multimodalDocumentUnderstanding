"""Chunk documents from a dataset directory and load embeddings into Redis Stack."""

import os
from pathlib import Path
from typing import Optional


_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_INDEX_NAME = "doc_chunks"
_VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension


def _get_index_schema():
    from redisvl.schema import IndexSchema
    return IndexSchema.from_dict({
        "index": {"name": _INDEX_NAME, "prefix": "doc"},
        "fields": [
            {"name": "text", "type": "text"},
            {"name": "doc_id", "type": "tag"},
            {"name": "doc_path", "type": "tag"},
            {"name": "page_index", "type": "numeric"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": _VECTOR_DIM,
                    "distance_metric": "cosine",
                    "algorithm": "hnsw",
                    "datatype": "float32",
                },
            },
        ],
    })


def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def _extract_text_from_image(image_path: str) -> str:
    """Best-effort text extraction: use pytesseract if available, else return empty string."""
    try:
        from PIL import Image
        import pytesseract
        return pytesseract.image_to_string(Image.open(image_path))
    except Exception:
        return Path(image_path).stem.replace("_", " ")  # fallback: filename as text


def index_dataset(dataset: str, data_dir: str, redis_url: str) -> dict:
    """Index all documents in data_dir into Redis Stack. Returns chunk count."""
    import redis as sync_redis
    from redisvl.index import SearchIndex
    from sentence_transformers import SentenceTransformer

    r = sync_redis.from_url(redis_url)
    schema = _get_index_schema()
    index = SearchIndex(schema, redis_client=r)

    # Drop existing index if it exists, then recreate
    try:
        index.delete(drop=True)
    except Exception:
        pass
    index.create(overwrite=True)

    model = SentenceTransformer(_EMBEDDING_MODEL)
    data_path = Path(data_dir)
    image_paths = list(data_path.rglob("*.png")) + list(data_path.rglob("*.jpg"))

    records = []
    for img_path in image_paths:
        doc_id = img_path.stem
        text = _extract_text_from_image(str(img_path))
        chunks = _chunk_text(text)
        embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            records.append({
                "id": f"{doc_id}:chunk:{i}",
                "doc_id": doc_id,
                "doc_path": str(img_path),
                "page_index": 0,
                "text": chunk,
                "embedding": emb,
            })

    if records:
        index.load(records)
    return {"chunks_indexed": len(records), "documents": len(image_paths)}
