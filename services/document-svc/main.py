import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis as sync_redis

import indexer
import search as search_module
from shared.observability import setup_tracing, setup_metrics, get_logger

app = FastAPI(title="document-svc", version="1.0")
setup_tracing("document-svc")
setup_metrics(app)
logger = get_logger("document-svc")

_REDIS_URL = os.getenv("REDIS_URL", "redis://redis-stack:6379")


def _get_redis() -> sync_redis.Redis:
    return sync_redis.from_url(_REDIS_URL)


class IndexRequest(BaseModel):
    dataset: str
    data_dir: str


@app.post("/documents/index")
def index_documents(req: IndexRequest):
    result = indexer.index_dataset(req.dataset, req.data_dir, _REDIS_URL)
    logger.info("Indexed dataset", extra={"dataset": req.dataset, "chunks": result.get("chunks_indexed")})
    return result


@app.delete("/documents/index")
def clear_index():
    r = _get_redis()
    keys = r.keys("doc:*")
    if keys:
        r.delete(*keys)
    try:
        r.execute_command("FT.DROPINDEX", "doc_chunks", "DD")
    except Exception:
        pass
    return {"deleted_keys": len(keys)}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5


class SearchResponse(BaseModel):
    chunks: list[dict]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    r = _get_redis()
    try:
        chunks = search_module.hybrid_search(
            query=req.query,
            top_k=req.top_k,
            alpha=req.alpha,
            redis=r,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return SearchResponse(chunks=chunks)


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    r = _get_redis()
    keys = r.keys(f"doc:{doc_id}:chunk:*")
    if not keys:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"doc_id": doc_id, "chunk_count": len(keys)}


@app.get("/health")
def health():
    return {"status": "ok"}
