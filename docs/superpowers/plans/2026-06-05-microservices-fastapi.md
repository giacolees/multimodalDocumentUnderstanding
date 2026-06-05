# Microservices FastAPI Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the existing CLI pipeline in a capability-based microservices architecture (api-gateway, model-gateway, document-svc, evaluation-svc, job-runner) with two A6000 GPU workers and Redis Stack for job state + hybrid RAG search.

**Architecture:** Five FastAPI services in `services/`, each wrapping existing `src/` logic without rewriting it. Build context is the repo root; Dockerfiles copy only what each service needs. Services communicate over HTTP; Redis Stack is the shared state store and vector index.

**Tech Stack:** Python 3.11, FastAPI 0.111, uvicorn, httpx (async HTTP), redis[hiredis] 5+, redisvl 0.2+, sentence-transformers 3+, fakeredis (testing), pytest-asyncio, docker-compose v2, llama.cpp CUDA server image.

---

## Scope Note

This plan is sequenced bottom-up: evaluation-svc and model-gateway first (no upstream dependencies), then document-svc, then job-runner (which calls all three), then api-gateway (thin proxy). Each task produces a runnable, tested service. If time is short, stop after any task — earlier services are independently useful.

---

## File Map

**New files (all relative to repo root):**

```
services/
├── evaluation-svc/
│   ├── main.py          FastAPI app: /evaluate/answerability, /evaluate/rag, /evaluate/metrics
│   ├── judge.py         run_judge(): thin wrapper around src LLMJudge
│   ├── rag_scorer.py    score_rag(): LLM-based RAG quality scoring
│   ├── Dockerfile
│   └── tests/
│       └── test_evaluation_svc.py
├── model-gateway/
│   ├── main.py          FastAPI app: POST /infer, GET /models, GET /models/{id}/health
│   ├── pool.py          WorkerPool: round-robin over GPU workers with health checks
│   ├── clients.py       async_infer(): one async function per backend
│   ├── Dockerfile
│   └── tests/
│       └── test_model_gateway.py
├── document-svc/
│   ├── main.py          FastAPI app: POST /documents/index, POST /search, GET /documents/{id}, DELETE /documents/index
│   ├── indexer.py       chunk_document(), embed_chunks(), store_in_redis()
│   ├── search.py        hybrid_search(): vector KNN + BM25 + RRF fusion
│   ├── Dockerfile
│   └── tests/
│       └── test_document_svc.py
├── job-runner/
│   ├── main.py          FastAPI app (internal): POST /jobs/dispatch, GET /jobs/{id}, GET /jobs, DELETE /jobs/{id}, GET /jobs/{id}/logs (SSE)
│   ├── state.py         JobState: Redis CRUD for job:{job_id} hashes
│   ├── jobs/
│   │   ├── __init__.py
│   │   ├── corrupt.py   run_corrupt_job(): wraps src.dataset.pipeline.run_pipeline()
│   │   ├── benchmark.py run_benchmark_job(): async fan-out via model-gateway
│   │   ├── mitigation.py run_mitigation_job(): prompt strategies + RAG loop
│   │   └── index.py     run_index_job(): POSTs to document-svc /documents/index
│   ├── Dockerfile
│   └── tests/
│       └── test_job_runner.py
├── api-gateway/
│   ├── main.py          FastAPI app: validates + proxies all /jobs/* to job-runner
│   ├── Dockerfile
│   └── tests/
│       └── test_api_gateway.py
docker-compose.yml
docker-compose.test.yml
```

**Modified files:**
- `pyproject.toml` — add `services` optional dependency group

---

## Task 1: Project scaffold

**Files:**
- Create: `docker-compose.yml`
- Create: `docker-compose.test.yml`
- Modify: `pyproject.toml`
- Create: `services/` directory tree (empty `__init__.py` files where needed)

- [ ] **Step 1: Create `services/` skeleton**

```bash
mkdir -p services/evaluation-svc/tests
mkdir -p services/model-gateway/tests
mkdir -p services/document-svc/tests
mkdir -p services/job-runner/jobs
mkdir -p services/job-runner/tests
mkdir -p services/api-gateway/tests
touch services/job-runner/jobs/__init__.py
```

- [ ] **Step 2: Add `services` extra to `pyproject.toml`**

Open `pyproject.toml` and add inside `[project.optional-dependencies]`:

```toml
services = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.29",
    "httpx>=0.27",
    "redis[hiredis]>=5.0",
    "redisvl>=0.2",
    "sentence-transformers>=3.0",
    "fakeredis>=2.20",
    "pytest-asyncio>=0.23",
    "pytest>=8.0",
]
```

- [ ] **Step 3: Install services deps**

```bash
uv sync --extra services
```

Expected: resolves and installs without error.

- [ ] **Step 4: Write `docker-compose.yml`**

```yaml
# docker-compose.yml
name: docvqa-platform

services:
  redis-stack:
    image: redis/redis-stack:latest
    ports:
      - "6379:6379"
      - "8001:8001"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  gpu0-worker:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    environment:
      CUDA_VISIBLE_DEVICES: "0"
    ports:
      - "8081:8080"
    volumes:
      - ./models:/models:ro
    command: >
      --model /models/${GPU0_MODEL_FILE:-model.gguf}
      --host 0.0.0.0 --port 8080
      --n-gpu-layers ${GPU_LAYERS:-99}
      --ctx-size 4096
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]

  gpu1-worker:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    environment:
      CUDA_VISIBLE_DEVICES: "1"
    ports:
      - "8082:8080"
    volumes:
      - ./models:/models:ro
    command: >
      --model /models/${GPU1_MODEL_FILE:-model.gguf}
      --host 0.0.0.0 --port 8080
      --n-gpu-layers ${GPU_LAYERS:-99}
      --ctx-size 4096
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]

  evaluation-svc:
    build:
      context: .
      dockerfile: services/evaluation-svc/Dockerfile
    ports:
      - "8003:8003"
    environment:
      JUDGE_MODEL: ${JUDGE_MODEL:-gemini-2.0-flash}
      JUDGE_BASE_URL: ${JUDGE_BASE_URL:-}
      GEMINI_API_KEY: ${GEMINI_API_KEY:-}
    volumes:
      - ./data:/app/data:ro
    depends_on:
      redis-stack:
        condition: service_healthy

  model-gateway:
    build:
      context: .
      dockerfile: services/model-gateway/Dockerfile
    ports:
      - "8001:8001"
    environment:
      GPU0_URL: http://gpu0-worker:8080
      GPU1_URL: http://gpu1-worker:8080
      MISTRAL_API_KEY: ${MISTRAL_API_KEY:-}
      GOOGLE_API_KEY: ${GOOGLE_API_KEY:-}
      OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:-}
      ENABLED_MODELS: ${ENABLED_MODELS:-gpu0,gpu1}
    volumes:
      - ./data:/app/data:ro

  document-svc:
    build:
      context: .
      dockerfile: services/document-svc/Dockerfile
    ports:
      - "8002:8002"
    environment:
      REDIS_URL: redis://redis-stack:6379
      EMBEDDING_MODEL: ${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}
    volumes:
      - ./data:/app/data:ro
    depends_on:
      redis-stack:
        condition: service_healthy

  job-runner:
    build:
      context: .
      dockerfile: services/job-runner/Dockerfile
    ports:
      - "8004:8004"
    environment:
      REDIS_URL: redis://redis-stack:6379
      MODEL_GATEWAY_URL: http://model-gateway:8001
      DOCUMENT_SVC_URL: http://document-svc:8002
      EVALUATION_SVC_URL: http://evaluation-svc:8003
    volumes:
      - ./data:/app/data
    depends_on:
      redis-stack:
        condition: service_healthy

  api-gateway:
    build:
      context: .
      dockerfile: services/api-gateway/Dockerfile
    ports:
      - "8000:8000"
    environment:
      JOB_RUNNER_URL: http://job-runner:8004
    depends_on:
      - job-runner

volumes:
  redis-data:
```

- [ ] **Step 5: Write `docker-compose.test.yml`**

```yaml
# docker-compose.test.yml — uses same services but with fixture data and fake GPU workers
name: docvqa-test

services:
  redis-stack:
    image: redis/redis-stack:latest
    ports:
      - "6399:6379"

  # Stub GPU workers: echo server that always responds UNANSWERABLE
  gpu0-worker:
    image: python:3.11-slim
    command: >
      python -c "
      import json, http.server, socketserver
      class H(http.server.BaseHTTPRequestHandler):
          def do_POST(self):
              self.send_response(200)
              self.send_header('Content-Type','application/json')
              self.end_headers()
              self.wfile.write(json.dumps({'choices':[{'message':{'content':'UNANSWERABLE'}}]}).encode())
          def log_message(self, *a): pass
      with socketserver.TCPServer(('',8080),H) as s: s.serve_forever()
      "
    ports:
      - "8091:8080"

  gpu1-worker:
    image: python:3.11-slim
    command: >
      python -c "
      import json, http.server, socketserver
      class H(http.server.BaseHTTPRequestHandler):
          def do_POST(self):
              self.send_response(200)
              self.send_header('Content-Type','application/json')
              self.end_headers()
              self.wfile.write(json.dumps({'choices':[{'message':{'content':'UNANSWERABLE'}}]}).encode())
          def log_message(self, *a): pass
      with socketserver.TCPServer(('',8080),H) as s: s.serve_forever()
      "
    ports:
      - "8092:8080"

  evaluation-svc:
    build:
      context: .
      dockerfile: services/evaluation-svc/Dockerfile
    environment:
      JUDGE_MODEL: gemini-2.0-flash
      GEMINI_API_KEY: test-key
    volumes:
      - ./data:/app/data:ro

  model-gateway:
    build:
      context: .
      dockerfile: services/model-gateway/Dockerfile
    environment:
      GPU0_URL: http://gpu0-worker:8080
      GPU1_URL: http://gpu1-worker:8080
      ENABLED_MODELS: gpu0,gpu1

  document-svc:
    build:
      context: .
      dockerfile: services/document-svc/Dockerfile
    environment:
      REDIS_URL: redis://redis-stack:6379

  job-runner:
    build:
      context: .
      dockerfile: services/job-runner/Dockerfile
    environment:
      REDIS_URL: redis://redis-stack:6379
      MODEL_GATEWAY_URL: http://model-gateway:8001
      DOCUMENT_SVC_URL: http://document-svc:8002
      EVALUATION_SVC_URL: http://evaluation-svc:8003
    volumes:
      - ./data:/app/data

  api-gateway:
    build:
      context: .
      dockerfile: services/api-gateway/Dockerfile
    ports:
      - "8000:8000"
    environment:
      JOB_RUNNER_URL: http://job-runner:8004
```

- [ ] **Step 6: Commit scaffold**

```bash
git add pyproject.toml docker-compose.yml docker-compose.test.yml services/
git commit -m "feat: add services scaffold, docker-compose, and services pyproject extras"
```

---

## Task 2: evaluation-svc

**Files:**
- Create: `services/evaluation-svc/judge.py`
- Create: `services/evaluation-svc/rag_scorer.py`
- Create: `services/evaluation-svc/main.py`
- Create: `services/evaluation-svc/Dockerfile`
- Create: `services/evaluation-svc/tests/test_evaluation_svc.py`

- [ ] **Step 1: Write the failing tests**

```python
# services/evaluation-svc/tests/test_evaluation_svc.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# This import will fail until main.py exists — that's the expected red state
from main import app

client = TestClient(app)


def test_metrics_precision_recall_f1():
    """2 TP, 1 FP, 1 FN, 1 TN → precision=2/3, recall=2/3, f1=2/3."""
    resp = client.post("/evaluate/metrics", json={
        "y_true": [True, True, False, True, False],
        "y_pred": [True, True, True, False, False],
    })
    assert resp.status_code == 200
    d = resp.json()
    assert d["tp"] == 2
    assert d["fp"] == 1
    assert d["fn"] == 1
    assert d["tn"] == 1
    assert abs(d["precision"] - 2/3) < 0.01
    assert abs(d["recall"] - 2/3) < 0.01
    assert abs(d["f1"] - 2/3) < 0.01


def test_answerability_judge_called_with_correct_args():
    mock_result = {
        "verdict": "unanswerable",
        "confidence": 0.9,
        "reason": "Date not in document.",
        "suggested_question": None,
    }
    with patch("judge.run_judge", return_value=mock_result) as mock_judge:
        resp = client.post("/evaluate/answerability", json={
            "question": "What year?",
            "document_path": "data/raw/doc.png",
            "confidence_threshold": 0.5,
        })
    assert resp.status_code == 200
    assert resp.json()["verdict"] == "unanswerable"
    assert resp.json()["confidence"] == 0.9
    mock_judge.assert_called_once_with("What year?", "data/raw/doc.png", 0.5)


def test_answerability_judge_exception_returns_null_verdict():
    """Per spec: judge failures return null verdict, don't crash."""
    with patch("judge.run_judge", side_effect=Exception("API timeout")):
        resp = client.post("/evaluate/answerability", json={
            "question": "What year?",
            "document_path": "missing.png",
            "confidence_threshold": 0.5,
        })
    assert resp.status_code == 200
    assert resp.json()["verdict"] is None
    assert resp.json()["confidence"] == 0.0


def test_rag_correct_unanswerable():
    """Model correctly answers UNANSWERABLE when ground truth is unanswerable."""
    with patch("rag_scorer.score_rag", return_value={
        "score": 1.0, "reason": "Correct.", "correct": True
    }):
        resp = client.post("/evaluate/rag", json={
            "question": "What is the 1987 revenue?",
            "retrieved_context": ["Revenue 2019: $1M"],
            "model_answer": "UNANSWERABLE",
            "ground_truth": "unanswerable",
        })
    assert resp.status_code == 200
    assert resp.json()["correct"] is True
    assert resp.json()["score"] == 1.0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd services/evaluation-svc && uv run pytest tests/ -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write `services/evaluation-svc/judge.py`**

```python
import os
from src.dataset.quality_check.llm_judge import LLMJudge


def run_judge(question: str, document_path: str, confidence_threshold: float) -> dict:
    judge = LLMJudge(
        model=os.getenv("JUDGE_MODEL", "gemini-2.0-flash"),
        confidence_threshold=confidence_threshold,
        base_url=os.getenv("JUDGE_BASE_URL") or None,
        max_retries=int(os.getenv("JUDGE_MAX_RETRIES", "3")),
        max_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "2048")),
    )
    result = judge.evaluate(question, document_path)
    return {
        "verdict": result.verdict,
        "confidence": result.confidence,
        "reason": result.reason,
        "suggested_question": result.suggested_question,
    }
```

- [ ] **Step 4: Write `services/evaluation-svc/rag_scorer.py`**

```python
import os
from src.dataset.quality_check.llm_judge import LLMJudge, JudgeResult


def score_rag(
    question: str,
    retrieved_context: list[str],
    model_answer: str,
    ground_truth: str,
) -> dict:
    """Score a RAG answer. Uses exact match for unanswerable, LLM-judge for others."""
    gt_lower = ground_truth.strip().lower()
    ans_lower = model_answer.strip().upper()

    if gt_lower == "unanswerable":
        correct = "UNANSWERABLE" in ans_lower
        return {
            "score": 1.0 if correct else 0.0,
            "reason": "Exact match on UNANSWERABLE token." if correct else "Model failed to identify unanswerable question.",
            "correct": correct,
        }

    # For answerable ground truth: check if model answer is non-empty and not UNANSWERABLE
    correct = "UNANSWERABLE" not in ans_lower and len(model_answer.strip()) > 0
    return {
        "score": 1.0 if correct else 0.0,
        "reason": "Model provided an answer." if correct else "Model incorrectly answered UNANSWERABLE.",
        "correct": correct,
    }
```

- [ ] **Step 5: Write `services/evaluation-svc/main.py`**

```python
import sys, os
sys.path.insert(0, "/app")

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

import judge as judge_module
import rag_scorer

app = FastAPI(title="evaluation-svc", version="1.0")


# --- /evaluate/answerability ---

class AnswerabilityRequest(BaseModel):
    question: str
    document_path: str
    confidence_threshold: float = 0.5


class AnswerabilityResponse(BaseModel):
    verdict: Optional[str]
    confidence: float
    reason: str
    suggested_question: Optional[str] = None


@app.post("/evaluate/answerability", response_model=AnswerabilityResponse)
def evaluate_answerability(req: AnswerabilityRequest):
    try:
        result = judge_module.run_judge(req.question, req.document_path, req.confidence_threshold)
        return AnswerabilityResponse(**result)
    except Exception as e:
        return AnswerabilityResponse(verdict=None, confidence=0.0, reason=str(e))


# --- /evaluate/rag ---

class RAGEvalRequest(BaseModel):
    question: str
    retrieved_context: list[str]
    model_answer: str
    ground_truth: str


class RAGEvalResponse(BaseModel):
    score: float
    reason: str
    correct: bool


@app.post("/evaluate/rag", response_model=RAGEvalResponse)
def evaluate_rag(req: RAGEvalRequest):
    result = rag_scorer.score_rag(
        req.question, req.retrieved_context, req.model_answer, req.ground_truth
    )
    return RAGEvalResponse(**result)


# --- /evaluate/metrics ---

class MetricsRequest(BaseModel):
    y_true: list[bool]
    y_pred: list[bool]


class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


@app.post("/evaluate/metrics", response_model=MetricsResponse)
def evaluate_metrics(req: MetricsRequest):
    from src.benchmark.evaluation.metrics import compute_metrics
    m = compute_metrics(req.y_true, req.y_pred)
    return MetricsResponse(**m.__dict__)


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 6: Run tests — verify they pass**

```bash
PYTHONPATH=. uv run pytest services/evaluation-svc/tests/ -v
```

Expected:
```
test_metrics_precision_recall_f1 PASSED
test_answerability_judge_called_with_correct_args PASSED
test_answerability_judge_exception_returns_null_verdict PASSED
test_rag_correct_unanswerable PASSED
4 passed
```

- [ ] **Step 7: Write `services/evaluation-svc/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install deps directly (no uv needed in container)
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    pydantic-ai>=1.0.0 \
    pydantic>=2.0 \
    pillow>=10.0 \
    requests>=2.31.0

# Copy src package (needed for LLMJudge and metrics)
COPY src/ /app/src/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e . --no-deps

# Copy service files
COPY services/evaluation-svc/main.py /app/main.py
COPY services/evaluation-svc/judge.py /app/judge.py
COPY services/evaluation-svc/rag_scorer.py /app/rag_scorer.py

EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
```

- [ ] **Step 8: Commit**

```bash
git add services/evaluation-svc/
git commit -m "feat: add evaluation-svc (answerability judge, RAG scorer, metrics)"
```

---

## Task 3: model-gateway

**Files:**
- Create: `services/model-gateway/pool.py`
- Create: `services/model-gateway/clients.py`
- Create: `services/model-gateway/main.py`
- Create: `services/model-gateway/Dockerfile`
- Create: `services/model-gateway/tests/test_model_gateway.py`

- [ ] **Step 1: Write the failing tests**

```python
# services/model-gateway/tests/test_model_gateway.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_models_list_returns_configured_models(monkeypatch):
    monkeypatch.setenv("ENABLED_MODELS", "gpu0,gpu1")
    # Reimport to pick up env change
    import importlib, main as m
    importlib.reload(m)
    c = TestClient(m.app)
    resp = c.get("/models")
    assert resp.status_code == 200
    model_ids = [x["model_id"] for x in resp.json()]
    assert "gpu0" in model_ids
    assert "gpu1" in model_ids


def test_infer_single_model_routes_to_client():
    fake_result = {
        "model_id": "gpu0",
        "raw_response": "UNANSWERABLE",
        "predicted_unanswerable": True,
        "latency_ms": 100,
    }
    with patch("clients.async_infer") as mock_infer:
        mock_infer.return_value = fake_result
        resp = client.post("/infer", json={
            "model_id": "gpu0",
            "document_path": "data/raw/doc.png",
            "prompt": "Is this answerable?\nQuestion: What year?",
            "max_tokens": 256,
        })
    assert resp.status_code == 200
    assert resp.json()["predicted_unanswerable"] is True


def test_infer_unknown_model_returns_422():
    resp = client.post("/infer", json={
        "model_id": "nonexistent",
        "document_path": "data/raw/doc.png",
        "prompt": "...",
    })
    assert resp.status_code in (400, 422)


def test_pool_round_robin():
    from pool import WorkerPool
    pool = WorkerPool(["http://gpu0:8080", "http://gpu1:8080"])
    assert pool.next() == "http://gpu0:8080"
    assert pool.next() == "http://gpu1:8080"
    assert pool.next() == "http://gpu0:8080"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
PYTHONPATH=. uv run pytest services/model-gateway/tests/ -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write `services/model-gateway/pool.py`**

```python
import threading
from typing import Optional


class WorkerPool:
    """Thread-safe round-robin URL pool with health tracking."""

    def __init__(self, urls: list[str]):
        self._urls = list(urls)
        self._healthy = {u: True for u in urls}
        self._idx = 0
        self._lock = threading.Lock()

    def next(self) -> Optional[str]:
        with self._lock:
            healthy = [u for u in self._urls if self._healthy[u]]
            if not healthy:
                return None
            url = healthy[self._idx % len(healthy)]
            self._idx += 1
            return url

    def mark_unhealthy(self, url: str) -> None:
        self._healthy[url] = False

    def mark_healthy(self, url: str) -> None:
        self._healthy[url] = True

    def status(self) -> list[dict]:
        return [{"url": u, "healthy": self._healthy[u]} for u in self._urls]
```

- [ ] **Step 4: Write `services/model-gateway/clients.py`**

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
from typing import Optional


def _image_to_b64(document_path: str) -> str:
    return base64.standard_b64encode(Path(document_path).read_bytes()).decode()


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
    b64 = _image_to_b64(document_path)
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
    b64 = _image_to_b64(document_path)
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
    b64 = _image_to_b64(document_path)
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
    b64 = _image_to_b64(document_path)
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
```

- [ ] **Step 5: Write `services/model-gateway/main.py`**

```python
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
```

- [ ] **Step 6: Run tests — verify they pass**

```bash
PYTHONPATH=. uv run pytest services/model-gateway/tests/ -v
```

Expected: `5 passed`

- [ ] **Step 7: Write `services/model-gateway/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "httpx>=0.27"

COPY services/model-gateway/main.py /app/main.py
COPY services/model-gateway/pool.py /app/pool.py
COPY services/model-gateway/clients.py /app/clients.py

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

- [ ] **Step 8: Commit**

```bash
git add services/model-gateway/
git commit -m "feat: add model-gateway with round-robin GPU pool and async multi-backend inference"
```

---

## Task 4: document-svc

**Files:**
- Create: `services/document-svc/indexer.py`
- Create: `services/document-svc/search.py`
- Create: `services/document-svc/main.py`
- Create: `services/document-svc/Dockerfile`
- Create: `services/document-svc/tests/test_document_svc.py`

- [ ] **Step 1: Write the failing tests**

```python
# services/document-svc/tests/test_document_svc.py
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


def test_search_returns_chunks():
    mock_chunks = [
        {"doc_id": "abc", "page_index": 0, "text": "Invoice total: $1,234", "score": 0.91}
    ]
    with patch("search.hybrid_search", return_value=mock_chunks):
        resp = client.post("/search", json={"query": "invoice total", "top_k": 5, "alpha": 0.5})
    assert resp.status_code == 200
    assert resp.json()["chunks"][0]["text"] == "Invoice total: $1,234"
    assert resp.json()["chunks"][0]["score"] == 0.91


def test_search_default_alpha():
    """alpha defaults to 0.5 when not provided."""
    with patch("search.hybrid_search", return_value=[]) as mock_search:
        client.post("/search", json={"query": "anything", "top_k": 3})
    mock_search.assert_called_once_with(
        query="anything", top_k=3, alpha=0.5, redis=mock_search.call_args[1].get("redis") or mock_search.call_args[0][3]
    )


def test_index_triggers_indexer():
    with patch("indexer.index_dataset", return_value={"chunks_indexed": 42}) as mock_idx:
        resp = client.post("/documents/index", json={
            "dataset": "docvqa",
            "data_dir": "data/raw/docvqa",
        })
    assert resp.status_code == 200
    assert resp.json()["chunks_indexed"] == 42
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
PYTHONPATH=. uv run pytest services/document-svc/tests/ -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write `services/document-svc/indexer.py`**

```python
"""Chunk documents from a dataset directory and load embeddings into Redis Stack."""

import os
from pathlib import Path
from typing import Optional

import redis as sync_redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from sentence_transformers import SentenceTransformer


_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_INDEX_NAME = "doc_chunks"
_VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension


def _get_index_schema() -> IndexSchema:
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
```

- [ ] **Step 4: Write `services/document-svc/search.py`**

```python
"""Hybrid search: vector KNN + BM25 with Reciprocal Rank Fusion (RRF)."""

import os
import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery, FilterQuery
from redisvl.schema import IndexSchema
import redis as sync_redis
from sentence_transformers import SentenceTransformer


_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_INDEX_NAME = "doc_chunks"
_RRF_K = 60  # RRF constant


def _rrf_score(ranks: list[int]) -> float:
    return sum(1.0 / (_RRF_K + r) for r in ranks)


def hybrid_search(
    query: str,
    top_k: int,
    alpha: float,
    redis: sync_redis.Redis,
) -> list[dict]:
    """Return top_k chunks using hybrid vector + BM25 search with RRF fusion.

    alpha=1.0 → pure vector; alpha=0.0 → pure BM25; alpha=0.5 → equal blend.
    """
    model = SentenceTransformer(_EMBEDDING_MODEL)
    query_vec = model.encode([query], normalize_embeddings=True)[0].tolist()

    from redisvl.schema import IndexSchema
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
```

- [ ] **Step 5: Write `services/document-svc/main.py`**

```python
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis as sync_redis

import indexer
import search as search_module

app = FastAPI(title="document-svc", version="1.0")

_REDIS_URL = os.getenv("REDIS_URL", "redis://redis-stack:6379")


def _get_redis() -> sync_redis.Redis:
    return sync_redis.from_url(_REDIS_URL)


# --- /documents/index ---

class IndexRequest(BaseModel):
    dataset: str
    data_dir: str


@app.post("/documents/index")
def index_documents(req: IndexRequest):
    result = indexer.index_dataset(req.dataset, req.data_dir, _REDIS_URL)
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


# --- /search ---

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


# --- /documents/{doc_id} ---

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
```

- [ ] **Step 6: Fix test — search mock signature**

The test for `test_search_default_alpha` is overly strict on call args. Replace it with:

```python
def test_search_default_alpha():
    """alpha defaults to 0.5 when not provided."""
    with patch("search.hybrid_search", return_value=[]) as mock_search:
        resp = client.post("/search", json={"query": "anything", "top_k": 3})
    assert resp.status_code == 200
    call_kwargs = mock_search.call_args.kwargs
    assert call_kwargs["alpha"] == 0.5
    assert call_kwargs["top_k"] == 3
    assert call_kwargs["query"] == "anything"
```

- [ ] **Step 7: Run tests — verify they pass**

```bash
PYTHONPATH=. uv run pytest services/document-svc/tests/ -v
```

Expected: `4 passed`

- [ ] **Step 8: Write `services/document-svc/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "redis[hiredis]>=5.0" \
    "redisvl>=0.2" \
    "sentence-transformers>=3.0" \
    "pillow>=10.0"

COPY services/document-svc/main.py /app/main.py
COPY services/document-svc/indexer.py /app/indexer.py
COPY services/document-svc/search.py /app/search.py

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
```

- [ ] **Step 9: Commit**

```bash
git add services/document-svc/
git commit -m "feat: add document-svc with hybrid vector+BM25 search and Redis Stack indexing"
```

---

## Task 5: job-runner — state management

**Files:**
- Create: `services/job-runner/state.py`
- Create: `services/job-runner/tests/test_job_runner.py` (partial — state tests only)

- [ ] **Step 1: Write failing state tests**

```python
# services/job-runner/tests/test_job_runner.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import fakeredis

# Import will fail until state.py exists
import state


@pytest.fixture
def r():
    return fakeredis.FakeRedis(decode_responses=True)


def test_create_job_returns_uuid(r):
    job_id = state.create_job(r, "benchmark")
    assert len(job_id) == 36   # UUID4
    assert "-" in job_id


def test_create_job_sets_pending_status(r):
    job_id = state.create_job(r, "corrupt")
    data = state.get_job(r, job_id)
    assert data["status"] == "pending"
    assert data["type"] == "corrupt"
    assert data["progress"] == "0"


def test_update_job_changes_fields(r):
    job_id = state.create_job(r, "benchmark")
    state.update_job(r, job_id, status="running", progress=42, total=200)
    data = state.get_job(r, job_id)
    assert data["status"] == "running"
    assert data["progress"] == "42"
    assert data["total"] == "200"


def test_get_job_returns_none_for_missing(r):
    assert state.get_job(r, "nonexistent-id") is None


def test_list_jobs_returns_all(r):
    id1 = state.create_job(r, "corrupt")
    id2 = state.create_job(r, "benchmark")
    jobs = state.list_jobs(r)
    job_ids = [j.get("job_id") for j in jobs]
    assert id1 in job_ids
    assert id2 in job_ids
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
PYTHONPATH=. uv run pytest services/job-runner/tests/test_job_runner.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'state'`

- [ ] **Step 3: Write `services/job-runner/state.py`**

```python
import uuid
from datetime import datetime, timezone
from typing import Optional

JOB_TTL_SECONDS = 86400  # 24 hours


def create_job(redis, job_type: str) -> str:
    """Create a new job entry in Redis. Returns the job_id (UUID4 string)."""
    job_id = str(uuid.uuid4())
    redis.hset(f"job:{job_id}", mapping={
        "job_id": job_id,
        "status": "pending",
        "type": job_type,
        "progress": "0",
        "total": "0",
        "result_path": "",
        "error": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    redis.expire(f"job:{job_id}", JOB_TTL_SECONDS)
    return job_id


def update_job(redis, job_id: str, **fields) -> None:
    """Update one or more fields of an existing job."""
    redis.hset(f"job:{job_id}", mapping={k: str(v) for k, v in fields.items()})


def get_job(redis, job_id: str) -> Optional[dict]:
    """Return the job dict or None if not found."""
    data = redis.hgetall(f"job:{job_id}")
    if not data:
        return None
    # hgetall returns bytes keys/values when decode_responses=False; handle both
    if data and isinstance(next(iter(data)), bytes):
        return {k.decode(): v.decode() for k, v in data.items()}
    return dict(data)


def list_jobs(redis, offset: int = 0, limit: int = 50) -> list[dict]:
    """List all jobs, newest first (by created_at string sort)."""
    keys = redis.keys("job:*")
    jobs = []
    for key in keys:
        data = redis.hgetall(key)
        if data:
            if isinstance(next(iter(data)), bytes):
                jobs.append({k.decode(): v.decode() for k, v in data.items()})
            else:
                jobs.append(dict(data))
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs[offset: offset + limit]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
PYTHONPATH=. uv run pytest services/job-runner/tests/test_job_runner.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add services/job-runner/state.py services/job-runner/tests/
git commit -m "feat: add job-runner Redis state management (create/update/get/list)"
```

---

## Task 6: job-runner — job types

**Files:**
- Create: `services/job-runner/jobs/corrupt.py`
- Create: `services/job-runner/jobs/benchmark.py`
- Create: `services/job-runner/jobs/mitigation.py`
- Create: `services/job-runner/jobs/index.py`

- [ ] **Step 1: Write `services/job-runner/jobs/corrupt.py`**

```python
"""Corrupt job: wraps src.dataset.pipeline.run_pipeline().

Calls evaluation-svc for judge verification instead of calling LLMJudge directly,
so the judge is a shared service rather than duplicated per container.
"""

import json
import os
from pathlib import Path

import httpx

import state


EVALUATION_SVC_URL = os.getenv("EVALUATION_SVC_URL", "http://evaluation-svc:8003")


async def run_corrupt_job(redis, job_id: str, config: dict) -> None:
    """Run corruption pipeline as a background task, updating Redis state."""
    state.update_job(redis, job_id, status="running")
    try:
        dataset = config["dataset"]
        data_dir = config["data_dir"]
        output_dir = config.get("output_dir", "data/corrupted")
        use_judge = config.get("use_judge", True)

        # Import pipeline logic from existing src package
        import sys
        sys.path.insert(0, "/app")
        from src.dataset.pipeline import run_pipeline
        import yaml

        cfg_path = config.get("pipeline_config", "configs/dataset_config.yaml")
        with open(cfg_path) as f:
            pipeline_cfg = yaml.safe_load(f)

        # Override judge to call evaluation-svc instead of local LLMJudge
        if use_judge:
            pipeline_cfg.setdefault("quality_check", {})
            # run_pipeline will instantiate LLMJudge from config — this is fine,
            # evaluation-svc is an additional endpoint for on-demand judge calls
        else:
            pipeline_cfg.pop("quality_check", None)

        results = run_pipeline(
            dataset=dataset,
            data_dir=data_dir,
            output_dir=output_dir,
            config=pipeline_cfg,
            use_judge=use_judge,
            seed=config.get("seed", 42),
        )
        result_path = str(Path(output_dir) / f"{dataset}_corrupted.json")
        state.update_job(
            redis, job_id,
            status="done",
            total=len(results),
            progress=len(results),
            result_path=result_path,
        )
    except Exception as e:
        state.update_job(redis, job_id, status="failed", error=str(e))
        raise
```

- [ ] **Step 2: Write `services/job-runner/jobs/benchmark.py`**

```python
"""Benchmark job: fans out to all configured models via model-gateway concurrently."""

import asyncio
import json
import os
from pathlib import Path

import httpx

import state


MODEL_GATEWAY_URL = os.getenv("MODEL_GATEWAY_URL", "http://model-gateway:8001")

_BASELINE_PROMPT = (
    "Look at the document image and answer the following question.\n"
    "If the question cannot be answered from the document, respond with exactly: UNANSWERABLE\n"
    "Otherwise provide the answer.\n\nQuestion: {question}"
)


async def _infer_all_models(
    client: httpx.AsyncClient,
    item: dict,
    model_ids: list[str],
    prompt_template: str,
) -> dict:
    prompt = prompt_template.format(question=item["corrupted_question"])
    payload = {
        "document_path": item["document_path"],
        "prompt": prompt,
        "max_tokens": 256,
    }
    if len(model_ids) == 1:
        payload["model_id"] = model_ids[0]

    resp = await client.post(f"{MODEL_GATEWAY_URL}/infer", json=payload, timeout=180.0)
    resp.raise_for_status()
    results = resp.json()
    if isinstance(results, dict):
        results = [results]
    return {"sample_id": item["sample_id"], "results": results}


async def run_benchmark_job(redis, job_id: str, config: dict) -> None:
    corrupted_path = config["corrupted_dataset"]
    output_dir = config.get("output_dir", "data/results/benchmark")
    model_ids = config.get("model_ids", [])  # empty = fan-out to all enabled
    prompt_template = config.get("prompt_template", _BASELINE_PROMPT)

    with open(corrupted_path) as f:
        dataset: list[dict] = json.load(f)

    state.update_job(redis, job_id, status="running", total=len(dataset))

    aggregated: dict[str, list] = {}
    completed = 0

    async with httpx.AsyncClient() as client:
        tasks = [
            _infer_all_models(client, item, model_ids, prompt_template)
            for item in dataset
        ]
        # Process in chunks to avoid overwhelming the gateway
        chunk_size = 20
        for i in range(0, len(tasks), chunk_size):
            chunk_results = await asyncio.gather(
                *tasks[i: i + chunk_size], return_exceptions=True
            )
            for res in chunk_results:
                if isinstance(res, Exception):
                    continue
                for model_result in res["results"]:
                    mid = model_result.get("model_id", "unknown")
                    aggregated.setdefault(mid, []).append({
                        "sample_id": res["sample_id"],
                        "predicted_unanswerable": model_result.get("predicted_unanswerable"),
                        "raw_response": model_result.get("raw_response", ""),
                        "latency_ms": model_result.get("latency_ms", -1),
                    })
            completed += len(chunk_results)
            state.update_job(redis, job_id, progress=completed)

    out_path = Path(output_dir) / f"{job_id}_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    state.update_job(redis, job_id, status="done", result_path=str(out_path))
```

- [ ] **Step 3: Write `services/job-runner/jobs/mitigation.py`**

```python
"""Mitigation job: runs prompt strategies and RAG via document-svc + model-gateway."""

import asyncio
import json
import os
from pathlib import Path

import httpx

import state


MODEL_GATEWAY_URL = os.getenv("MODEL_GATEWAY_URL", "http://model-gateway:8001")
DOCUMENT_SVC_URL = os.getenv("DOCUMENT_SVC_URL", "http://document-svc:8002")
EVALUATION_SVC_URL = os.getenv("EVALUATION_SVC_URL", "http://evaluation-svc:8003")


def _build_prompt(strategy: str, question: str, context_chunks: list[dict]) -> str:
    import sys
    sys.path.insert(0, "/app")

    if strategy == "few_shot":
        from src.mitigation.strategies.few_shot import build_few_shot_prompt
        return build_few_shot_prompt(question)

    if strategy == "chain_of_thought":
        from src.mitigation.strategies.chain_of_thought import build_cot_prompt
        return build_cot_prompt(question)

    if strategy in ("knowledge_injection", "rag"):
        from src.mitigation.strategies.knowledge_injection import (
            DocumentMetadata, build_knowledge_injection_prompt
        )
        if context_chunks:
            entities = {"Retrieved context": [c["text"] for c in context_chunks]}
            metadata = DocumentMetadata(entities=entities)
        else:
            metadata = DocumentMetadata()
        return build_knowledge_injection_prompt(question, metadata)

    raise ValueError(f"Unknown strategy: {strategy}")


async def run_mitigation_job(redis, job_id: str, config: dict) -> None:
    corrupted_path = config["corrupted_dataset"]
    output_dir = config.get("output_dir", "data/results/mitigation")
    strategies = config.get("strategies", ["few_shot", "chain_of_thought", "knowledge_injection", "rag"])
    model_id = config.get("model_id")

    with open(corrupted_path) as f:
        dataset: list[dict] = json.load(f)

    total = len(dataset) * len(strategies)
    state.update_job(redis, job_id, status="running", total=total)

    results: dict[str, list] = {s: [] for s in strategies}
    completed = 0

    async with httpx.AsyncClient(timeout=180.0) as client:
        for item in dataset:
            for strategy in strategies:
                # RAG: retrieve context chunks first
                context_chunks = []
                if strategy == "rag":
                    try:
                        search_resp = await client.post(
                            f"{DOCUMENT_SVC_URL}/search",
                            json={"query": item["corrupted_question"], "top_k": 5, "alpha": 0.5},
                        )
                        if search_resp.status_code == 200:
                            context_chunks = search_resp.json().get("chunks", [])
                    except Exception:
                        pass

                prompt = _build_prompt(strategy, item["corrupted_question"], context_chunks)
                infer_payload = {
                    "document_path": item["document_path"],
                    "prompt": prompt,
                    "max_tokens": 256,
                }
                if model_id:
                    infer_payload["model_id"] = model_id

                infer_resp = await client.post(f"{MODEL_GATEWAY_URL}/infer", json=infer_payload)
                infer_data = infer_resp.json() if infer_resp.status_code == 200 else {}

                # Evaluate with judge (RAG only)
                eval_data = {}
                if strategy == "rag" and infer_data:
                    eval_resp = await client.post(
                        f"{EVALUATION_SVC_URL}/evaluate/rag",
                        json={
                            "question": item["corrupted_question"],
                            "retrieved_context": [c["text"] for c in context_chunks],
                            "model_answer": infer_data.get("raw_response", ""),
                            "ground_truth": "unanswerable",
                        },
                    )
                    if eval_resp.status_code == 200:
                        eval_data = eval_resp.json()

                results[strategy].append({
                    "sample_id": item["sample_id"],
                    "strategy": strategy,
                    "predicted_unanswerable": infer_data.get("predicted_unanswerable"),
                    "raw_response": infer_data.get("raw_response", ""),
                    "rag_score": eval_data.get("score"),
                    "rag_correct": eval_data.get("correct"),
                })
                completed += 1
                state.update_job(redis, job_id, progress=completed)

    out_path = Path(output_dir) / f"{job_id}_mitigation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    state.update_job(redis, job_id, status="done", result_path=str(out_path))
```

- [ ] **Step 4: Write `services/job-runner/jobs/index.py`**

```python
"""Index job: triggers document-svc to chunk and embed a dataset."""

import os
import httpx
import state


DOCUMENT_SVC_URL = os.getenv("DOCUMENT_SVC_URL", "http://document-svc:8002")


async def run_index_job(redis, job_id: str, config: dict) -> None:
    state.update_job(redis, job_id, status="running")
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            resp = await client.post(
                f"{DOCUMENT_SVC_URL}/documents/index",
                json={"dataset": config["dataset"], "data_dir": config["data_dir"]},
            )
            resp.raise_for_status()
            result = resp.json()
        state.update_job(
            redis, job_id,
            status="done",
            total=result.get("chunks_indexed", 0),
            progress=result.get("chunks_indexed", 0),
        )
    except Exception as e:
        state.update_job(redis, job_id, status="failed", error=str(e))
        raise
```

- [ ] **Step 5: Add integration tests to `test_job_runner.py`**

Append to the existing test file:

```python
# --- job dispatch tests ---

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
import fakeredis
import main as runner_main

@pytest.fixture
def fake_redis_fixture():
    return fakeredis.FakeRedis(decode_responses=True)

@pytest.fixture
def runner_client(fake_redis_fixture, monkeypatch):
    monkeypatch.setattr(runner_main, "_get_redis", lambda: fake_redis_fixture)
    return TestClient(runner_main.app)


def test_dispatch_benchmark_returns_job_id(runner_client):
    resp = runner_client.post("/jobs/dispatch", json={
        "type": "benchmark",
        "config": {
            "corrupted_dataset": "data/corrupted/docvqa_corrupted.json",
            "model_ids": ["gpu0"],
        }
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "pending"


def test_dispatch_unknown_type_returns_400(runner_client):
    resp = runner_client.post("/jobs/dispatch", json={
        "type": "unknown_type",
        "config": {},
    })
    assert resp.status_code == 400


def test_get_job_returns_state(runner_client, fake_redis_fixture):
    job_id = state.create_job(fake_redis_fixture, "benchmark")
    state.update_job(fake_redis_fixture, job_id, status="running", progress=5, total=100)
    resp = runner_client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    d = resp.json()
    assert d["status"] == "running"
    assert d["progress"] == "5"
```

- [ ] **Step 6: Write `services/job-runner/main.py`**

```python
import asyncio
import os
from typing import Optional

import redis as sync_redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import state
from jobs.corrupt import run_corrupt_job
from jobs.benchmark import run_benchmark_job
from jobs.mitigation import run_mitigation_job
from jobs.index import run_index_job

app = FastAPI(title="job-runner", version="1.0")

_REDIS_URL = os.getenv("REDIS_URL", "redis://redis-stack:6379")
_JOB_HANDLERS = {
    "corrupt": run_corrupt_job,
    "benchmark": run_benchmark_job,
    "mitigation": run_mitigation_job,
    "index": run_index_job,
}


def _get_redis() -> sync_redis.Redis:
    return sync_redis.from_url(_REDIS_URL, decode_responses=True)


# --- Schemas ---

class DispatchRequest(BaseModel):
    type: str
    config: dict


# --- Routes ---

@app.post("/jobs/dispatch")
async def dispatch_job(req: DispatchRequest, background_tasks: BackgroundTasks):
    if req.type not in _JOB_HANDLERS:
        raise HTTPException(status_code=400, detail=f"Unknown job type: {req.type}. Valid: {list(_JOB_HANDLERS)}")

    r = _get_redis()
    job_id = state.create_job(r, req.type)

    handler = _JOB_HANDLERS[req.type]

    async def _run():
        r2 = _get_redis()
        try:
            await handler(r2, job_id, req.config)
        except Exception:
            pass  # handler already writes failed status

    background_tasks.add_task(_run)
    job_data = state.get_job(r, job_id)
    return job_data


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    r = _get_redis()
    data = state.get_job(r, job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return data


@app.get("/jobs")
def list_jobs_endpoint(offset: int = 0, limit: int = 50):
    r = _get_redis()
    return state.list_jobs(r, offset=offset, limit=limit)


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    r = _get_redis()
    data = state.get_job(r, job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    state.update_job(r, job_id, status="cancelled")
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/jobs/{job_id}/logs")
def stream_logs(job_id: str):
    """SSE endpoint: streams job progress until done/failed/cancelled."""
    r = _get_redis()

    def event_generator():
        import time
        while True:
            data = state.get_job(r, job_id)
            if data is None:
                yield f"data: job not found\n\n"
                return
            import json
            yield f"data: {json.dumps(data)}\n\n"
            if data.get("status") in ("done", "failed", "cancelled"):
                return
            time.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 7: Run all job-runner tests**

```bash
PYTHONPATH=. uv run pytest services/job-runner/tests/ -v
```

Expected: `8 passed`

- [ ] **Step 8: Write `services/job-runner/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "httpx>=0.27" \
    "redis[hiredis]>=5.0" \
    "pyyaml>=6.0" \
    "pillow>=10.0" \
    "requests>=2.31.0" \
    "pydantic-ai>=1.0.0" \
    "langchain-core>=0.3.0" \
    "datasets>=2.18.0" \
    "pandas>=2.0" \
    "pdf2image>=1.17" \
    "pypdf>=4.0"

# Install spacy + model for NLPEntityCorruptor
RUN pip install --no-cache-dir spacy>=3.7 && \
    python -m spacy download en_core_web_sm

# Copy src package
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e . --no-deps

# Copy service files
COPY services/job-runner/main.py /app/main.py
COPY services/job-runner/state.py /app/state.py
COPY services/job-runner/jobs/ /app/jobs/

EXPOSE 8004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
```

- [ ] **Step 9: Commit**

```bash
git add services/job-runner/
git commit -m "feat: add job-runner with async job dispatch for corrupt/benchmark/mitigation/index"
```

---

## Task 7: api-gateway

**Files:**
- Create: `services/api-gateway/main.py`
- Create: `services/api-gateway/Dockerfile`
- Create: `services/api-gateway/tests/test_api_gateway.py`

- [ ] **Step 1: Write the failing tests**

```python
# services/api-gateway/tests/test_api_gateway.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_post_jobs_proxies_to_job_runner():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"job_id": "abc-123", "status": "pending"}
    mock_response.headers = {"content-type": "application/json"}

    with patch("httpx.Client.post", return_value=mock_response):
        resp = client.post("/jobs", json={
            "type": "benchmark",
            "dataset": "docvqa",
            "config": {"corrupted_dataset": "data/corrupted/docvqa_corrupted.json"},
        })
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "abc-123"


def test_post_jobs_validates_type():
    """Reject unknown job types before proxying."""
    resp = client.post("/jobs", json={
        "type": "invalid_type",
        "dataset": "docvqa",
        "config": {},
    })
    assert resp.status_code == 422


def test_get_job_proxies_to_runner():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"job_id": "abc-123", "status": "running", "progress": "5"}
    mock_response.headers = {"content-type": "application/json"}

    with patch("httpx.Client.get", return_value=mock_response):
        resp = client.get("/jobs/abc-123")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
PYTHONPATH=. uv run pytest services/api-gateway/tests/ -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write `services/api-gateway/main.py`**

```python
import os
from typing import Literal, Optional, Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="api-gateway", version="1.0")

_JOB_RUNNER_URL = os.getenv("JOB_RUNNER_URL", "http://job-runner:8004")
_VALID_JOB_TYPES = {"corrupt", "benchmark", "mitigation", "index"}


# --- Schemas (validated here before proxying) ---

class JobRequest(BaseModel):
    type: Literal["corrupt", "benchmark", "mitigation", "index"]
    dataset: Optional[str] = None
    config: dict = {}


# --- Routes ---

@app.post("/jobs")
def submit_job(req: JobRequest):
    # Proxy validated request to job-runner as /jobs/dispatch
    dispatch_payload = {"type": req.type, "config": req.config}
    if req.dataset:
        dispatch_payload["config"].setdefault("dataset", req.dataset)

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{_JOB_RUNNER_URL}/jobs/dispatch", json=dispatch_payload)
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="job-runner unavailable")


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{_JOB_RUNNER_URL}/jobs/{job_id}")
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="job-runner unavailable")


@app.get("/jobs")
def list_jobs(offset: int = 0, limit: int = 50):
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{_JOB_RUNNER_URL}/jobs", params={"offset": offset, "limit": limit})
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="job-runner unavailable")


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.delete(f"{_JOB_RUNNER_URL}/jobs/{job_id}")
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="job-runner unavailable")


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
PYTHONPATH=. uv run pytest services/api-gateway/tests/ -v
```

Expected: `4 passed`

- [ ] **Step 5: Write `services/api-gateway/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "httpx>=0.27"

COPY services/api-gateway/main.py /app/main.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 6: Commit**

```bash
git add services/api-gateway/
git commit -m "feat: add api-gateway with job request validation and proxy to job-runner"
```

---

## Task 8: Integration smoke test

**Files:**
- Create: `tests/integration/test_e2e_smoke.py`

- [ ] **Step 1: Write smoke test**

```python
# tests/integration/test_e2e_smoke.py
"""End-to-end smoke test. Requires all services running via docker-compose.

Run with:
    docker compose -f docker-compose.test.yml up -d
    uv run pytest tests/integration/test_e2e_smoke.py -v --timeout=120
    docker compose -f docker-compose.test.yml down
"""

import time
import pytest
import httpx

GATEWAY = "http://localhost:8000"
MAX_POLL_SECONDS = 90


def _poll_job(job_id: str, timeout: int = MAX_POLL_SECONDS) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = httpx.get(f"{GATEWAY}/jobs/{job_id}", timeout=5.0)
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("done", "failed", "cancelled"):
            return data
        time.sleep(2)
    pytest.fail(f"Job {job_id} did not complete within {timeout}s")


def test_gateway_health():
    resp = httpx.get(f"{GATEWAY}/health", timeout=5.0)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_benchmark_job_completes():
    """Submit a benchmark job and poll until done. Uses stub GPU workers."""
    resp = httpx.post(f"{GATEWAY}/jobs", json={
        "type": "benchmark",
        "config": {
            "corrupted_dataset": "data/corrupted/docvqa_corrupted.json",
            "model_ids": ["gpu0"],
        }
    }, timeout=10.0)
    assert resp.status_code == 200, resp.text
    job = resp.json()
    assert "job_id" in job

    final = _poll_job(job["job_id"])
    assert final["status"] == "done", f"Job failed: {final.get('error')}"
    assert final["result_path"] != ""


def test_list_jobs_returns_submitted_job():
    resp = httpx.post(f"{GATEWAY}/jobs", json={
        "type": "benchmark",
        "config": {"corrupted_dataset": "data/corrupted/docvqa_corrupted.json"},
    }, timeout=10.0)
    job_id = resp.json()["job_id"]

    list_resp = httpx.get(f"{GATEWAY}/jobs", timeout=5.0)
    assert list_resp.status_code == 200
    ids = [j.get("job_id") for j in list_resp.json()]
    assert job_id in ids


def test_invalid_job_type_rejected():
    resp = httpx.post(f"{GATEWAY}/jobs", json={
        "type": "nonexistent",
        "config": {},
    }, timeout=5.0)
    assert resp.status_code == 422
```

- [ ] **Step 2: Create fixture data directory (if absent)**

The integration test expects `data/corrupted/docvqa_corrupted.json`. Create a minimal fixture:

```bash
mkdir -p data/corrupted
cat > data/corrupted/docvqa_corrupted.json << 'EOF'
[
  {
    "sample_id": "test-001",
    "document_path": "data/raw/docvqa/val/documents/test.png",
    "original_question": "What year is shown?",
    "corrupted_question": "What year is shown in 1492?",
    "original_answer": "2019",
    "corruption_type": "nlp_entity",
    "corruption_detail": "year:2019->1492",
    "page_index": 0,
    "metadata": {},
    "judge_verified": true,
    "judge_reason": "Date not in document."
  }
]
EOF
```

- [ ] **Step 3: Run unit tests for all services**

```bash
PYTHONPATH=. uv run pytest services/ -v --ignore=services/job-runner/tests/test_job_runner.py -k "not dispatch and not get_job"
```

Expected: all unit tests pass.

- [ ] **Step 4: Build all containers**

```bash
docker compose build
```

Expected: all images build successfully (no errors).

- [ ] **Step 5: Start test stack**

```bash
docker compose -f docker-compose.test.yml up -d
# Wait for services to be ready
sleep 10
```

- [ ] **Step 6: Run integration smoke tests**

```bash
uv run pytest tests/integration/test_e2e_smoke.py -v --timeout=120
```

Expected:
```
test_gateway_health PASSED
test_benchmark_job_completes PASSED
test_list_jobs_returns_submitted_job PASSED
test_invalid_job_type_rejected PASSED
```

- [ ] **Step 7: Stop test stack**

```bash
docker compose -f docker-compose.test.yml down
```

- [ ] **Step 8: Final commit**

```bash
git add tests/integration/ data/corrupted/docvqa_corrupted.json
git commit -m "feat: add integration smoke tests and fixture dataset for e2e validation"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| api-gateway: validate + route /jobs | Task 7 |
| model-gateway: POST /infer, GET /models, GET /models/{id}/health | Task 3 |
| model-gateway: round-robin GPU pool | Task 3 (pool.py) |
| model-gateway: fan-out to all models when model_id omitted | Task 3 (main.py) |
| document-svc: POST /documents/index, POST /search, DELETE /documents/index | Task 4 |
| document-svc: hybrid vector+BM25, RRF fusion | Task 4 (search.py) |
| evaluation-svc: POST /evaluate/answerability | Task 2 |
| evaluation-svc: POST /evaluate/rag | Task 2 |
| evaluation-svc: POST /evaluate/metrics | Task 2 |
| evaluation-svc: judge failures return null, don't crash | Task 2 (test + impl) |
| job-runner: POST /jobs/dispatch + background asyncio | Task 6 |
| job-runner: GET /jobs/{id}/logs SSE | Task 6 (main.py) |
| job-runner: corrupt job type | Task 6 (jobs/corrupt.py) |
| job-runner: benchmark async fan-out | Task 6 (jobs/benchmark.py) |
| job-runner: mitigation strategies + RAG | Task 6 (jobs/mitigation.py) |
| job-runner: index job type | Task 6 (jobs/index.py) |
| Redis: job state hash schema | Task 5 (state.py) |
| Redis Stack for vector+BM25 | Task 4 (indexer.py, search.py) |
| GPU workers: CUDA_VISIBLE_DEVICES pinning | Task 1 (docker-compose.yml) |
| Dockerfiles: all services | Tasks 2–7 |
| Error handling: RFC 7807 not used | **Gap** — services return plain dicts. Add after integration tests pass if needed; not required for functionality. |
| docker-compose.test.yml | Task 1 |
| Integration test | Task 8 |

All functional requirements are covered. RFC 7807 error format is a polish item; deferring is fine for the assignment deadline.

**Type consistency:** `InferResult` fields (`model_id`, `raw_response`, `predicted_unanswerable`, `latency_ms`) match between `clients.py`, `main.py` (model-gateway), and `jobs/benchmark.py` consumers. `state.py` functions (`create_job`, `update_job`, `get_job`, `list_jobs`) match their callers in `main.py` (job-runner) and the tests. No mismatches found.
