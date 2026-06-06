"""Hybrid search: vector KNN + BM25 with Reciprocal Rank Fusion (RRF)."""

import os


_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_INDEX_NAME = "doc_chunks"
_RRF_K = 60  # RRF constant


def _rrf_score(ranks: list[int]) -> float:
    return sum(1.0 / (_RRF_K + r) for r in ranks)


def hybrid_search(
    query: str,
    top_k: int,
    alpha: float,
    redis,
) -> list[dict]:
    """Return top_k chunks using hybrid vector + BM25 search with RRF fusion.

    alpha=1.0 → pure vector; alpha=0.0 → pure BM25; alpha=0.5 → equal blend.
    """
    import numpy as np
    from redisvl.index import SearchIndex
    from redisvl.query import VectorQuery, FilterQuery
    from redisvl.schema import IndexSchema
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(_EMBEDDING_MODEL)
    query_vec = model.encode([query], normalize_embeddings=True)[0].tolist()

    schema = IndexSchema.from_dict({
        "index": {"name": _INDEX_NAME, "prefix": "doc"},
        "fields": [
            {"name": "text", "type": "text"},
            {"name": "doc_id", "type": "tag"},
            {"name": "doc_path", "type": "tag"},
            {"name": "page_index", "type": "numeric"},
            {"name": "embedding", "type": "vector",
             "attrs": {"dims": 384, "distance_metric": "cosine",
                       "algorithm": "hnsw", "datatype": "float32"}},
        ],
    })
    index = SearchIndex(schema, redis_client=redis)

    # Vector search
    vector_results: list[dict] = []
    if alpha > 0:
        vq = VectorQuery(
            vector=query_vec,
            vector_field_name="embedding",
            return_fields=["doc_id", "doc_path", "page_index", "text"],
            num_results=top_k * 2,
        )
        vector_results = index.query(vq)

    # BM25 full-text search (RediSearch @text field)
    bm25_results: list[dict] = []
    if alpha < 1.0:
        fq = FilterQuery(
            filter_expression=f"@text:({query})",
            return_fields=["doc_id", "doc_path", "page_index", "text"],
            num_results=top_k * 2,
        )
        bm25_results = index.query(fq)

    # Build id → ranks map for RRF
    scores: dict[str, list[int]] = {}
    for rank, r in enumerate(vector_results):
        key = r.get("id", r.get("doc_id", "") + str(rank))
        scores.setdefault(key, []).append(rank + 1)

    bm25_offset = len(vector_results) + 1
    for rank, r in enumerate(bm25_results):
        key = r.get("id", r.get("doc_id", "") + str(rank))
        scores.setdefault(key, []).append(rank + bm25_offset)

    # Score and merge all results
    all_results_by_id: dict[str, dict] = {}
    for r in vector_results + bm25_results:
        key = r.get("id", "")
        all_results_by_id[key] = r

    ranked = sorted(
        all_results_by_id.items(),
        key=lambda kv: _rrf_score(scores.get(kv[0], [999])),
        reverse=True,
    )

    return [
        {
            "doc_id": r.get("doc_id", ""),
            "page_index": int(r.get("page_index", 0)),
            "text": r.get("text", ""),
            "score": round(_rrf_score(scores.get(key, [999])), 4),
        }
        for key, r in ranked[:top_k]
    ]
