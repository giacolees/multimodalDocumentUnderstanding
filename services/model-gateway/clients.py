"""Async inference clients for each model backend.

All backends receive a base64-encoded PNG and a prompt string.
All return a dict matching InferResponse shape.
"""

import asyncio
import base64
import os
import time
from pathlib import Path
from typing import Optional


async def _image_to_b64(document_path: str) -> str:
    data = await asyncio.to_thread(Path(document_path).read_bytes)
    return base64.standard_b64encode(data).decode()


def _parse_unanswerable(text: str) -> bool:
    upper = text.upper()
    if "UNANSWERABLE" in upper:
        return True
    phrases = ["cannot be answered", "not in the document", "no information",
               "not mentioned", "not found", "cannot answer", "not provided"]
    return any(p.upper() in upper for p in phrases)


async def async_infer(
    model_id: str,
    document_path: str,
    prompt: str,
    max_tokens: int = 256,
    gpu_pool=None,        # WorkerPool instance, used when model_id in ("gpu0","gpu1")
) -> dict:
    t0 = time.monotonic()

    if model_id in ("gpu0", "gpu1"):
        result = await _infer_llama_cpp(model_id, document_path, prompt, max_tokens, gpu_pool)
    elif model_id == "mistral":
        result = await _infer_mistral(document_path, prompt, max_tokens)
    elif model_id == "google":
        result = await _infer_google(document_path, prompt, max_tokens)
    elif model_id == "openrouter":
        result = await _infer_openrouter(document_path, prompt, max_tokens)
    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    latency_ms = int((time.monotonic() - t0) * 1000)
    raw = result["raw_response"]
    return {
        "model_id": model_id,
        "raw_response": raw,
        "predicted_unanswerable": _parse_unanswerable(raw),
        "latency_ms": latency_ms,
    }


async def _infer_llama_cpp(model_id: str, document_path: str, prompt: str, max_tokens: int, pool) -> dict:
    import httpx
    url = pool.next() if pool else os.getenv(f"{model_id.upper()}_URL", "http://localhost:8080")
    if url is None:
        raise RuntimeError("No healthy GPU workers available")
    b64 = await _image_to_b64(document_path)
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return {"raw_response": raw}


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
    raw = resp.json()["choices"][0]["message"]["content"]
    return {"raw_response": raw}


async def _infer_google(document_path: str, prompt: str, max_tokens: int) -> dict:
    import httpx
    api_key = os.environ["GOOGLE_API_KEY"]
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
    raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return {"raw_response": raw}


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
    raw = resp.json()["choices"][0]["message"]["content"]
    return {"raw_response": raw}
