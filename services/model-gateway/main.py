import asyncio
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pool import WorkerPool
import clients

app = FastAPI(title="model-gateway", version="1.0")

# llama.cpp pool — comma-separated URLs from env
_LLAMA_URLS_RAW = os.getenv("LLAMA_URLS", "")
_llama_pool = WorkerPool([u.strip() for u in _LLAMA_URLS_RAW.split(",") if u.strip()])

# vLLM pool — comma-separated URLs from env
_VLLM_URLS_RAW = os.getenv("VLLM_URLS", "")
_vllm_pool = WorkerPool([u.strip() for u in _VLLM_URLS_RAW.split(",") if u.strip()])

_ENABLED = set(os.getenv("ENABLED_MODELS", "llama,vllm").split(","))
_ALL_MODELS = ["llama", "vllm", "mistral", "google", "openrouter"]


class InferRequest(BaseModel):
    model_id: Optional[str] = None
    document_path: str
    prompt: str
    max_tokens: int = 256


@app.post("/infer")
async def infer(req: InferRequest):
    if req.model_id is not None and req.model_id not in _ALL_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {req.model_id}")

    targets = [req.model_id] if req.model_id else [m for m in _ALL_MODELS if m in _ENABLED]

    tasks = [
        clients.async_infer(
            mid, req.document_path, req.prompt, req.max_tokens,
            llama_pool=_llama_pool, vllm_pool=_vllm_pool,
        )
        for mid in targets
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for mid, res in zip(targets, results):
        if isinstance(res, Exception):
            out.append({"model_id": mid, "error": str(res)})
        else:
            out.append(res)

    return out[0] if req.model_id else out


@app.get("/models")
def list_models():
    return [
        {
            "model_id": m,
            "enabled": m in _ENABLED,
            "pool_status": (
                _llama_pool.status() if m == "llama"
                else _vllm_pool.status() if m == "vllm"
                else None
            ),
        }
        for m in _ALL_MODELS
    ]


@app.get("/models/{model_id}/health")
async def model_health(model_id: str):
    if model_id not in _ALL_MODELS:
        raise HTTPException(status_code=404, detail="Unknown model")
    if model_id not in ("llama", "vllm"):
        return {"model_id": model_id, "healthy": True, "note": "API-based model"}

    pool = _llama_pool if model_id == "llama" else _vllm_pool
    import httpx
    results = []
    for entry in pool.status():
        url = entry["url"]
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                resp = await c.get(f"{url}/health")
            healthy = resp.status_code == 200
        except Exception:
            healthy = False
        if healthy:
            pool.mark_healthy(url)
        else:
            pool.mark_unhealthy(url)
        results.append({"url": url, "healthy": healthy})
    return {"model_id": model_id, "workers": results}


@app.get("/health")
def health():
    return {"status": "ok"}
