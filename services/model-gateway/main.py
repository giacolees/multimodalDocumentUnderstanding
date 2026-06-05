import asyncio
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pool import WorkerPool
import clients

app = FastAPI(title="model-gateway", version="1.0")

# Initialise GPU pool from env at startup
_GPU_URLS = {
    "gpu0": os.getenv("GPU0_URL", "http://gpu0-worker:8080"),
    "gpu1": os.getenv("GPU1_URL", "http://gpu1-worker:8080"),
}
_gpu_pool = WorkerPool(list(_GPU_URLS.values()))

_ENABLED = set(os.getenv("ENABLED_MODELS", "gpu0,gpu1").split(","))
_ALL_MODELS = ["gpu0", "gpu1", "mistral", "google", "openrouter"]


# --- Schemas ---

class InferRequest(BaseModel):
    model_id: Optional[str] = None   # None = fan-out to all enabled models
    document_path: str
    prompt: str
    max_tokens: int = 256


class InferResult(BaseModel):
    model_id: str
    raw_response: str
    predicted_unanswerable: bool
    latency_ms: int


# --- Routes ---

@app.post("/infer")
async def infer(req: InferRequest):
    if req.model_id is not None and req.model_id not in _ALL_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {req.model_id}")

    targets = [req.model_id] if req.model_id else [m for m in _ALL_MODELS if m in _ENABLED]

    tasks = [
        clients.async_infer(mid, req.document_path, req.prompt, req.max_tokens, _gpu_pool)
        for mid in targets
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for mid, res in zip(targets, results):
        if isinstance(res, Exception):
            out.append({"model_id": mid, "error": str(res)})
        else:
            out.append(res)

    # Single model → return dict; fan-out → return list
    return out[0] if req.model_id else out


@app.get("/models")
def list_models():
    return [
        {
            "model_id": m,
            "enabled": m in _ENABLED,
            "pool_status": _gpu_pool.status() if m in ("gpu0", "gpu1") else None,
        }
        for m in _ALL_MODELS
    ]


@app.get("/models/{model_id}/health")
async def model_health(model_id: str):
    if model_id not in _ALL_MODELS:
        raise HTTPException(status_code=404, detail="Unknown model")
    if model_id not in ("gpu0", "gpu1"):
        return {"model_id": model_id, "healthy": True, "note": "API-based model"}
    url = _GPU_URLS[model_id]
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{url}/health")
        healthy = resp.status_code == 200
    except Exception:
        healthy = False
    if healthy:
        _gpu_pool.mark_healthy(url)
    else:
        _gpu_pool.mark_unhealthy(url)
    return {"model_id": model_id, "healthy": healthy, "url": url}


@app.get("/health")
def health():
    return {"status": "ok"}
