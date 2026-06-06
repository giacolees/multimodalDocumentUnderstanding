# vLLM Backend + Symmetric Local Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `gpu0`/`gpu1` → a single `llama` pool, add a parallel `vllm` pool, and add vLLM worker containers — both backends configured symmetrically via `LLAMA_URLS`/`VLLM_URLS` env vars.

**Architecture:** Two `WorkerPool` instances (`_llama_pool`, `_vllm_pool`) replace the old `_GPU_URLS` dict. A shared `_infer_local` function handles both backends (identical OpenAI-compatible API). Docker Compose gains `llama-worker-0/1` (renamed) and `vllm-worker-0/1` (new).

**Tech Stack:** Python 3.11, FastAPI, httpx, docker-compose v2, `vllm/vllm-openai:latest`, `ghcr.io/ggerganov/llama.cpp:server-cuda`.

---

## File Map

**Modified:**
- `services/model-gateway/clients.py` — rename `_infer_llama_cpp` → `_infer_local`; update `async_infer` signature
- `services/model-gateway/main.py` — replace `_GPU_URLS`/`_gpu_pool` with `_llama_pool`+`_vllm_pool`; update `_ALL_MODELS`, routes
- `services/model-gateway/tests/test_model_gateway.py` — update existing tests; add vLLM tests
- `docker-compose.yml` — rename GPU workers; add vLLM workers; update model-gateway env
- `docker-compose.test.yml` — rename stub GPU workers; add stub vLLM worker

---

## Task 1: Refactor clients.py

**Files:**
- Modify: `services/model-gateway/clients.py`

Current `_infer_llama_cpp(model_id, document_path, prompt, max_tokens, pool)` takes `model_id` but only uses `pool`. Both llama.cpp and vLLM expose identical `/v1/chat/completions`. This task merges them into one `_infer_local` function and updates `async_infer` to accept named pool kwargs.

- [ ] **Step 1: Write the failing test**

```python
# Append to services/model-gateway/tests/test_model_gateway.py

def test_infer_llama_routes_to_local_client():
    fake_result = {
        "model_id": "llama",
        "raw_response": "UNANSWERABLE",
        "predicted_unanswerable": True,
        "latency_ms": 100,
    }
    with patch("clients.async_infer", return_value=fake_result):
        resp = client.post("/infer", json={
            "model_id": "llama",
            "document_path": "data/raw/doc.png",
            "prompt": "Is this answerable?",
            "max_tokens": 256,
        })
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "llama"


def test_infer_vllm_routes_to_local_client():
    fake_result = {
        "model_id": "vllm",
        "raw_response": "UNANSWERABLE",
        "predicted_unanswerable": True,
        "latency_ms": 80,
    }
    with patch("clients.async_infer", return_value=fake_result):
        resp = client.post("/infer", json={
            "model_id": "vllm",
            "document_path": "data/raw/doc.png",
            "prompt": "Is this answerable?",
        })
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "vllm"


def test_infer_vllm_no_workers_returns_error_in_fanout():
    with patch("clients.async_infer",
               side_effect=RuntimeError("No healthy workers available in pool")):
        resp = client.post("/infer", json={
            "model_id": "vllm",
            "document_path": "data/raw/doc.png",
            "prompt": "...",
        })
    assert resp.status_code == 200
    assert "error" in resp.json()
```

- [ ] **Step 2: Run — verify the new tests fail (main.py still has gpu0/gpu1)**

```bash
PYTHONPATH=. uv run pytest services/model-gateway/tests/test_model_gateway.py::test_infer_llama_routes_to_local_client -v
```

Expected: `FAILED` — `model_id: "llama"` returns 400 (unknown model).

- [ ] **Step 3: Rewrite `services/model-gateway/clients.py`**

Replace the entire file with:

```python
"""Async inference clients for each model backend.

All backends receive a base64-encoded PNG and a prompt string.
All return a dict matching InferResponse shape.
"""

import asyncio
import base64
import os
import time
from pathlib import Path


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
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
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

    latency_ms = int((time.monotonic() - t0) * 1000)
    raw = result["raw_response"]
    return {
        "model_id": model_id,
        "raw_response": raw,
        "predicted_unanswerable": _parse_unanswerable(raw),
        "latency_ms": latency_ms,
    }
```

- [ ] **Step 4: Rewrite `services/model-gateway/main.py`**

Replace the entire file with:

```python
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
```

- [ ] **Step 5: Update existing tests and verify all pass**

Replace the full content of `services/model-gateway/tests/test_model_gateway.py` with:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_models_list_returns_configured_models(monkeypatch):
    monkeypatch.setenv("ENABLED_MODELS", "llama,vllm")
    import importlib, main as m
    importlib.reload(m)
    c = TestClient(m.app)
    resp = c.get("/models")
    assert resp.status_code == 200
    model_ids = [x["model_id"] for x in resp.json()]
    assert "llama" in model_ids
    assert "vllm" in model_ids


def test_infer_llama_routes_to_local_client():
    fake_result = {
        "model_id": "llama",
        "raw_response": "UNANSWERABLE",
        "predicted_unanswerable": True,
        "latency_ms": 100,
    }
    with patch("clients.async_infer", return_value=fake_result):
        resp = client.post("/infer", json={
            "model_id": "llama",
            "document_path": "data/raw/doc.png",
            "prompt": "Is this answerable?",
            "max_tokens": 256,
        })
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "llama"
    assert resp.json()["predicted_unanswerable"] is True


def test_infer_vllm_routes_to_local_client():
    fake_result = {
        "model_id": "vllm",
        "raw_response": "UNANSWERABLE",
        "predicted_unanswerable": True,
        "latency_ms": 80,
    }
    with patch("clients.async_infer", return_value=fake_result):
        resp = client.post("/infer", json={
            "model_id": "vllm",
            "document_path": "data/raw/doc.png",
            "prompt": "Is this answerable?",
        })
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "vllm"


def test_infer_vllm_no_workers_returns_error_in_fanout():
    with patch("clients.async_infer",
               side_effect=RuntimeError("No healthy workers available in pool")):
        resp = client.post("/infer", json={
            "model_id": "vllm",
            "document_path": "data/raw/doc.png",
            "prompt": "...",
        })
    assert resp.status_code == 200
    assert "error" in resp.json()


def test_infer_unknown_model_returns_400():
    resp = client.post("/infer", json={
        "model_id": "nonexistent",
        "document_path": "data/raw/doc.png",
        "prompt": "...",
    })
    assert resp.status_code in (400, 422)


def test_pool_round_robin():
    from pool import WorkerPool
    pool = WorkerPool(["http://worker-a:8080", "http://worker-b:8080"])
    assert pool.next() == "http://worker-a:8080"
    assert pool.next() == "http://worker-b:8080"
    assert pool.next() == "http://worker-a:8080"
```

- [ ] **Step 6: Run all model-gateway tests — verify 7 passed**

```bash
PYTHONPATH=. uv run pytest services/model-gateway/tests/ -v
```

Expected:
```
test_health PASSED
test_models_list_returns_configured_models PASSED
test_infer_llama_routes_to_local_client PASSED
test_infer_vllm_routes_to_local_client PASSED
test_infer_vllm_no_workers_returns_error_in_fanout PASSED
test_infer_unknown_model_returns_400 PASSED
test_pool_round_robin PASSED
7 passed
```

- [ ] **Step 7: Commit**

```bash
git add services/model-gateway/clients.py \
        services/model-gateway/main.py \
        services/model-gateway/tests/test_model_gateway.py
git commit -m "feat: symmetric llama/vllm pools — rename gpu0/gpu1 to llama, add vllm backend"
```

---

## Task 2: Update docker-compose.yml

**Files:**
- Modify: `docker-compose.yml`

Rename the two GPU worker services, add two vLLM worker services, and update model-gateway's environment and `depends_on`.

- [ ] **Step 1: Rename `gpu0-worker` → `llama-worker-0`**

In `docker-compose.yml`, find the `gpu0-worker:` service block and replace it with:

```yaml
  llama-worker-0:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    environment:
      CUDA_VISIBLE_DEVICES: "0"
    ports:
      - "8081:8080"
    volumes:
      - ./models:/models:ro
    command: >
      --model /models/${LLAMA0_MODEL_FILE:-model.gguf}
      --host 0.0.0.0 --port 8080
      --n-gpu-layers ${GPU_LAYERS:-99}
      --ctx-size 4096
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 12
      start_period: 60s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
```

- [ ] **Step 2: Rename `gpu1-worker` → `llama-worker-1`**

Find the `gpu1-worker:` service block and replace it with:

```yaml
  llama-worker-1:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    environment:
      CUDA_VISIBLE_DEVICES: "1"
    ports:
      - "8082:8080"
    volumes:
      - ./models:/models:ro
    command: >
      --model /models/${LLAMA1_MODEL_FILE:-model.gguf}
      --host 0.0.0.0 --port 8080
      --n-gpu-layers ${GPU_LAYERS:-99}
      --ctx-size 4096
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 12
      start_period: 60s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
```

- [ ] **Step 3: Add `vllm-worker-0` and `vllm-worker-1` after the llama workers**

```yaml
  vllm-worker-0:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      CUDA_VISIBLE_DEVICES: "0"
      HF_HOME: /models/hf_cache
      HF_TOKEN: ${HF_TOKEN:-}
    ports:
      - "8083:8000"
    volumes:
      - ./models:/models
    command: >
      --model ${VLLM_MODEL:-Qwen/Qwen2-VL-7B-Instruct}
      --host 0.0.0.0 --port 8000
      --dtype auto
      --max-model-len ${VLLM_MAX_LEN:-4096}
      --limit-mm-per-prompt image=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      timeout: 5s
      retries: 20
      start_period: 120s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]

  vllm-worker-1:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      CUDA_VISIBLE_DEVICES: "1"
      HF_HOME: /models/hf_cache
      HF_TOKEN: ${HF_TOKEN:-}
    ports:
      - "8084:8000"
    volumes:
      - ./models:/models
    command: >
      --model ${VLLM_MODEL_1:-${VLLM_MODEL:-Qwen/Qwen2-VL-7B-Instruct}}
      --host 0.0.0.0 --port 8000
      --dtype auto
      --max-model-len ${VLLM_MAX_LEN:-4096}
      --limit-mm-per-prompt image=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      timeout: 5s
      retries: 20
      start_period: 120s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
```

- [ ] **Step 4: Update model-gateway environment and `depends_on`**

Find the `model-gateway:` service and replace its `environment` and `depends_on` blocks with:

```yaml
    environment:
      LLAMA_URLS: http://llama-worker-0:8080,http://llama-worker-1:8080
      VLLM_URLS: http://vllm-worker-0:8000,http://vllm-worker-1:8000
      MISTRAL_API_KEY: ${MISTRAL_API_KEY:-}
      GOOGLE_API_KEY: ${GOOGLE_API_KEY:-}
      OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:-}
      ENABLED_MODELS: ${ENABLED_MODELS:-llama,vllm}
    depends_on:
      llama-worker-0:
        condition: service_healthy
      llama-worker-1:
        condition: service_healthy
```

- [ ] **Step 5: Validate compose syntax**

```bash
docker compose config --quiet
```

Expected: no output (valid YAML).

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: rename gpu workers to llama-worker-{0,1}, add vllm-worker-{0,1} in docker-compose"
```

---

## Task 3: Update docker-compose.test.yml

**Files:**
- Modify: `docker-compose.test.yml`

The test compose uses stub Python echo servers for GPU workers. Rename them to match the new service names and add a stub vLLM worker.

- [ ] **Step 1: Rename `gpu0-worker` → `llama-worker-0` in the test compose**

Find the first stub worker block and replace it with:

```yaml
  llama-worker-0:
    image: python:3.11-slim
    command: >
      python -c "
      import json, http.server, socketserver
      class H(http.server.BaseHTTPRequestHandler):
          def do_POST(self):
              length = int(self.headers.get('Content-Length', 0))
              self.rfile.read(length)
              self.send_response(200)
              self.send_header('Content-Type','application/json')
              self.end_headers()
              self.wfile.write(json.dumps({'choices':[{'message':{'content':'UNANSWERABLE'}}]}).encode())
          def log_message(self, *a): pass
      with socketserver.TCPServer(('',8080),H) as s: s.serve_forever()
      "
    ports:
      - "8091:8080"
```

- [ ] **Step 2: Rename `gpu1-worker` → `llama-worker-1` in the test compose**

Find the second stub worker block and replace it with:

```yaml
  llama-worker-1:
    image: python:3.11-slim
    command: >
      python -c "
      import json, http.server, socketserver
      class H(http.server.BaseHTTPRequestHandler):
          def do_POST(self):
              length = int(self.headers.get('Content-Length', 0))
              self.rfile.read(length)
              self.send_response(200)
              self.send_header('Content-Type','application/json')
              self.end_headers()
              self.wfile.write(json.dumps({'choices':[{'message':{'content':'UNANSWERABLE'}}]}).encode())
          def log_message(self, *a): pass
      with socketserver.TCPServer(('',8080),H) as s: s.serve_forever()
      "
    ports:
      - "8092:8080"
```

- [ ] **Step 3: Add `vllm-worker-0` stub (listens on port 8000)**

```yaml
  vllm-worker-0:
    image: python:3.11-slim
    command: >
      python -c "
      import json, http.server, socketserver
      class H(http.server.BaseHTTPRequestHandler):
          def do_GET(self):
              self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
          def do_POST(self):
              length = int(self.headers.get('Content-Length', 0))
              self.rfile.read(length)
              self.send_response(200)
              self.send_header('Content-Type','application/json')
              self.end_headers()
              self.wfile.write(json.dumps({'choices':[{'message':{'content':'UNANSWERABLE'}}]}).encode())
          def log_message(self, *a): pass
      with socketserver.TCPServer(('',8000),H) as s: s.serve_forever()
      "
    ports:
      - "8085:8000"
```

- [ ] **Step 4: Update model-gateway env in test compose**

Find the `model-gateway:` service in `docker-compose.test.yml` and update its `environment` block:

```yaml
    environment:
      LLAMA_URLS: http://llama-worker-0:8080,http://llama-worker-1:8080
      VLLM_URLS: http://vllm-worker-0:8000
      ENABLED_MODELS: llama,vllm
```

- [ ] **Step 5: Validate test compose syntax**

```bash
docker compose -f docker-compose.test.yml config --quiet
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.test.yml
git commit -m "feat: update test compose — rename stub GPU workers, add stub vllm-worker-0"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Rename `gpu0`/`gpu1` → `llama` pool | Task 1 (clients.py, main.py, tests) |
| Add `vllm` pool via `VLLM_URLS` | Task 1 (main.py) |
| `_infer_local` shared by both backends | Task 1 (clients.py) |
| `async_infer` takes `llama_pool`, `vllm_pool` kwargs | Task 1 (clients.py) |
| `GET /models` reports both pool statuses | Task 1 (main.py) |
| `GET /models/{id}/health` checks all workers in pool | Task 1 (main.py) |
| Rename `gpu0-worker`/`gpu1-worker` → `llama-worker-0/1` | Task 2 |
| Add `vllm-worker-0`/`vllm-worker-1` to prod compose | Task 2 |
| `LLAMA_URLS`/`VLLM_URLS` in model-gateway env | Task 2 |
| Rename stub workers in test compose | Task 3 |
| Add stub `vllm-worker-0` to test compose | Task 3 |
| `ENABLED_MODELS` defaults to `llama,vllm` | Task 1 (main.py) |
| `HF_TOKEN`, `VLLM_MODEL`, `VLLM_MODEL_1`, `VLLM_MAX_LEN` env vars | Task 2 |
| 7 model-gateway tests pass | Task 1 |

All spec requirements covered. No gaps.

**Type consistency:** `async_infer(model_id, document_path, prompt, max_tokens, llama_pool, vllm_pool)` used consistently in `clients.py` definition and `main.py` call site. `_infer_local(document_path, prompt, max_tokens, pool)` called correctly from both `llama` and `vllm` branches.
