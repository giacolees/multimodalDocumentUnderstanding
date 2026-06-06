"""Hybrid search: vector KNN + BM25 with Reciprocal Rank Fusion (RRF)."""

import os


_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_INDEX_NAME = "doc_chunks"
_RRF_K = 60  # RRF constant


def _rrf_score_weighted(vector_ranks: list[int], bm25_ranks: list[int], alpha: float) -> float:
    """Weighted RRF: alpha weights vector contribution, (1-alpha) weights BM25."""
    vector_score = sum(1.0 / (_RRF_K + r) for r in vector_ranks) * alpha
    bm25_score = sum(1.0 / (_RRF_K + r) for r in bm25_ranks) * (1 - alpha)
    return vector_score + bm25_score


def _bm25_search(redis_client, query_text: str, top_k: int) -> list[dict]:
    """Full-text BM25 search via RediSearch FT.SEARCH."""
    try:
        from redis.commands.search.query import Query as RedisQuery
        q = (
            RedisQuery(query_text)
            .return_fields("doc_id", "text", "page_index")
            .paging(0, top_k)
        )
        results = redis_client.ft("doc_chunks").search(q)
        return [
            {
                "id": doc.id,
                "doc_id": getattr(doc, "doc_id", ""),
                "text": getattr(doc, "text", ""),
                "page_index": getattr(doc, "page_index", 0),
            }
            for doc in results.docs
        ]
    except Exception:
        return []


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
    from redisvl.query import VectorQuery
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

    # BM25 full-text search via FT.SEARCH (proper BM25 scoring)
    bm25_results: list[dict] = []
    if alpha < 1.0:
        bm25_results = _bm25_search(redis, query, top_k * 2)

    # Build per-source rank maps (each starting at 1 independently)
    vector_rank_map: dict[str, list[int]] = {}
    for rank, r in enumerate(vector_results):
        key = r.get("id", r.get("doc_id", "") + str(rank))
        vector_rank_map.setdefault(key, []).append(rank + 1)

    bm25_rank_map: dict[str, list[int]] = {}
    for rank, r in enumerate(bm25_results):
        key = r.get("id", r.get("doc_id", "") + str(rank))
        bm25_rank_map.setdefault(key, []).append(rank + 1)  # independent rank from 1

    # Merge all result keys
    all_keys = set(vector_rank_map) | set(bm25_rank_map)

    # Score and merge all results
    all_results_by_id: dict[str, dict] = {}
    for r in vector_results + bm25_results:
        key = r.get("id", "")
        all_results_by_id[key] = r

    ranked = sorted(
        [(k, all_results_by_id[k]) for k in all_keys if k in all_results_by_id],
        key=lambda kv: _rrf_score_weighted(
            vector_rank_map.get(kv[0], []),
            bm25_rank_map.get(kv[0], []),
            alpha,
        ),
        reverse=True,
    )

    return [
        {
            "doc_id": r.get("doc_id", ""),
            "page_index": int(r.get("page_index", 0)),
            "text": r.get("text", ""),
            "score": round(_rrf_score_weighted(
                vector_rank_map.get(key, []),
                bm25_rank_map.get(key, []),
                alpha,
            ), 4),
        }
        for key, r in ranked[:top_k]
    ]
