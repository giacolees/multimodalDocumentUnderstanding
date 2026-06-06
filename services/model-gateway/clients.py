"""Async inference clients for each model backend.

All backends receive a base64-encoded PNG and a prompt string.
All return a dict matching InferResponse shape.
"""

import asyncio
import base64
import os
import time
from pathlib import Path

from prometheus_client import Counter, Histogram

_INFER_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference round-trip latency per model",
    labelnames=["model_id"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 180],
)
_INFER_ERRORS = Counter(
    "inference_errors_total",
    "Inference failures by model and error type",
    labelnames=["model_id", "error_type"],
)


async def _image_to_b64(document_path: str) -> str:
    data = await asyncio.to_thread(Path(document_path).read_bytes)
    return base64.standard_b64encode(data).decode()


def _parse_unanswerable(text: str) -> bool:
    upper = text.upper()
    if "UNANSWERABLE" in upper:
        return True
    phrases = [
        "cannot be answered", "not in the document", "no information",
        "not mentioned", "not found", "cannot answer", "not provided",
    ]
    return any(p.upper() in upper for p in phrases)


async def _infer_local(document_path: str, prompt: str, max_tokens: int, pool) -> dict:
    """Shared client for llama.cpp and vLLM — both expose OpenAI-compatible /v1/chat/completions."""
    import httpx
    url = pool.next()
    if url is None:
        raise RuntimeError("No healthy workers available in pool")
    b64 = await _image_to_b64(document_path)
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(f"{url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
    except Exception:
        pool.mark_unhealthy(url)
        raise
    return {"raw_response": resp.json()["choices"][0]["message"]["content"]}


async def _infer_mistral(document_path: str, prompt: str, max_tokens: int) -> dict:
    import httpx
    api_key = os.environ["MISTRAL_API_KEY"]
    b64 = await _image_to_b64(document_path)
    payload = {
        "model": os.getenv("MISTRAL_MODEL_ID", "pixtral-12b-2409"),
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
    return {"raw_response": resp.json()["choices"][0]["message"]["content"]}


async def _infer_google(document_path: str, prompt: str, max_tokens: int) -> dict:
    import httpx
    api_key = os.environ["GEMINI_API_KEY"]
    b64 = await _image_to_b64(document_path)
    model_id = os.getenv("GOOGLE_MODEL_ID", "gemini-2.0-flash")
    payload = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/png", "data": b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.0},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
    return {"raw_response": resp.json()["candidates"][0]["content"]["parts"][0]["text"]}


async def _infer_openrouter(document_path: str, prompt: str, max_tokens: int) -> dict:
    import httpx
    api_key = os.environ["OPENROUTER_API_KEY"]
    b64 = await _image_to_b64(document_path)
    model_id = os.getenv("OPENROUTER_MODEL_ID", "google/gemini-2.0-flash-exp")
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
    return {"raw_response": resp.json()["choices"][0]["message"]["content"]}


async def async_infer(
    model_id: str,
    document_path: str,
    prompt: str,
    max_tokens: int = 256,
    llama_pool=None,
    vllm_pool=None,
) -> dict:
    t0 = time.monotonic()
    try:
        if model_id == "llama":
            result = await _infer_local(document_path, prompt, max_tokens, llama_pool)
        elif model_id == "vllm":
            result = await _infer_local(document_path, prompt, max_tokens, vllm_pool)
        elif model_id == "mistral":
            result = await _infer_mistral(document_path, prompt, max_tokens)
        elif model_id == "google":
            result = await _infer_google(document_path, prompt, max_tokens)
        elif model_id == "openrouter":
            result = await _infer_openrouter(document_path, prompt, max_tokens)
        else:
            raise ValueError(f"Unknown model_id: {model_id}")
    except Exception as e:
        _INFER_ERRORS.labels(model_id=model_id, error_type=type(e).__name__).inc()
        raise

    latency_s = time.monotonic() - t0
    _INFER_LATENCY.labels(model_id=model_id).observe(latency_s)
    raw = result["raw_response"]
    return {
        "model_id": model_id,
        "raw_response": raw,
        "predicted_unanswerable": _parse_unanswerable(raw),
        "latency_ms": int(latency_s * 1000),
    }
