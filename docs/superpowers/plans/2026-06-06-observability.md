# Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add distributed tracing (OTel → Jaeger), Prometheus metrics (inference latency, pool health, job counters), and structured JSON logging to the 5-service FastAPI stack.

**Architecture:** A single `services/shared/observability.py` module provides `setup_tracing(service_name)`, `setup_metrics(app)`, and `get_logger(name)`. Each service calls these two lines at startup. Custom metrics are added to `model-gateway/clients.py`, `model-gateway/main.py`, and `job-runner/state.py`. Two new Docker containers (Jaeger all-in-one, Prometheus) are added to `docker-compose.yml`.

**Tech Stack:** Python 3.11, opentelemetry-sdk 1.20+, opentelemetry-instrumentation-fastapi, opentelemetry-instrumentation-httpx, opentelemetry-exporter-otlp-proto-grpc, prometheus-fastapi-instrumentator 6.1+, prometheus-client 0.19+, jaegertracing/all-in-one, prom/prometheus.

---

## File Map

**New files:**
- `services/shared/__init__.py` — empty, makes `shared` a package
- `services/shared/observability.py` — `setup_tracing`, `setup_metrics`, `get_logger`
- `services/shared/tests/test_observability.py` — unit tests for the shared module
- `observability/prometheus.yml` — Prometheus scrape config

**Modified files:**
- `pyproject.toml` — add OTel + Prometheus deps to `services` extra
- `docker-compose.yml` — add `jaeger`, `prometheus` services; add `OTEL_EXPORTER_OTLP_ENDPOINT` + `LOG_LEVEL` to all 5 app services
- `services/evaluation-svc/Dockerfile` — add OTel+Prometheus pip installs + shared module copy
- `services/model-gateway/Dockerfile` — same
- `services/document-svc/Dockerfile` — same
- `services/job-runner/Dockerfile` — same
- `services/api-gateway/Dockerfile` — same
- `services/evaluation-svc/main.py` — add `setup_tracing` + `setup_metrics` calls
- `services/model-gateway/main.py` — add `setup_tracing` + `setup_metrics` + `_POOL_HEALTHY` gauge + lifespan
- `services/document-svc/main.py` — add `setup_tracing` + `setup_metrics`
- `services/job-runner/main.py` — add `setup_tracing` + `setup_metrics` + job_id span
- `services/api-gateway/main.py` — add `setup_tracing` + `setup_metrics`
- `services/model-gateway/clients.py` — add `_INFER_LATENCY` histogram + `_INFER_ERRORS` counter
- `services/job-runner/state.py` — add `_JOBS` counter, increment on terminal status

---

## Task 1: Shared observability module + pyproject.toml

**Files:**
- Create: `services/shared/__init__.py`
- Create: `services/shared/observability.py`
- Create: `services/shared/tests/__init__.py`
- Create: `services/shared/tests/test_observability.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add OTel + Prometheus deps to pyproject.toml**

Open `pyproject.toml`. Inside the `services = [...]` list, append:

```toml
    "opentelemetry-sdk>=1.20",
    "opentelemetry-instrumentation-fastapi>=0.41b0",
    "opentelemetry-exporter-otlp-proto-grpc>=1.20",
    "opentelemetry-instrumentation-httpx>=0.41b0",
    "prometheus-fastapi-instrumentator>=6.1",
    "prometheus-client>=0.19",
```

- [ ] **Step 2: Install new deps**

```bash
uv sync --extra services
```

Expected: resolves and installs without error. `opentelemetry-sdk` and `prometheus-fastapi-instrumentator` appear in output.

- [ ] **Step 3: Write failing tests**

```python
# services/shared/tests/test_observability.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import logging
import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_get_logger_returns_logger():
    from shared.observability import get_logger
    logger = get_logger("test-service")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test-service"


def test_get_logger_emits_json(capsys):
    from shared.observability import get_logger
    logger = get_logger("test-json")
    logger.info("hello world", extra={"job_id": "abc-123"})
    captured = capsys.readouterr()
    record = json.loads(captured.err)
    assert record["message"] == "hello world"
    assert record["job_id"] == "abc-123"
    assert record["level"] == "INFO"


def test_setup_tracing_does_not_raise_when_jaeger_unreachable():
    """OTel BatchSpanProcessor drops spans silently — service must not crash."""
    from shared.observability import setup_tracing
    # Should not raise even with a bad endpoint
    setup_tracing("test-service")


def test_setup_metrics_exposes_metrics_endpoint():
    from shared.observability import setup_metrics
    app = FastAPI()
    setup_metrics(app)
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "python_info" in resp.text  # default prometheus_client metric
```

- [ ] **Step 4: Run tests — verify they fail**

```bash
PYTHONPATH=. uv run pytest services/shared/tests/test_observability.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'shared'`

- [ ] **Step 5: Create `services/shared/__init__.py`**

```bash
mkdir -p services/shared/tests
touch services/shared/__init__.py
touch services/shared/tests/__init__.py
```

- [ ] **Step 6: Write `services/shared/observability.py`**

```python
"""Shared observability setup for all FastAPI services.

Usage in each service's main.py:
    from shared.observability import setup_tracing, setup_metrics, get_logger
    setup_tracing("service-name")
    setup_metrics(app)
    logger = get_logger("service-name")
"""

import json
import logging
import os


def setup_tracing(service_name: str) -> None:
    """Configure OTel SDK to export traces to Jaeger via OTLP gRPC.

    Safe to call when Jaeger is unreachable — BatchSpanProcessor drops spans
    silently; the service continues running without tracing.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
    except Exception as e:
        logging.getLogger(service_name).warning(
            "Observability setup failed — tracing disabled: %s", e
        )


def setup_metrics(app) -> None:
    """Attach prometheus-fastapi-instrumentator to the FastAPI app.

    Exposes GET /metrics in Prometheus text format.
    """
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def get_logger(name: str) -> logging.Logger:
    """Return a logger that emits structured JSON records to stderr."""

    class _JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            _skip = {
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "levelname", "levelno", "lineno",
                "message", "module", "msecs", "msg", "name", "pathname",
                "process", "processName", "relativeCreated", "stack_info",
                "thread", "threadName",
            }
            for key, val in record.__dict__.items():
                if key not in _skip:
                    payload[key] = val
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            return json.dumps(payload)

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    return logger
```

- [ ] **Step 7: Run tests — verify they pass**

```bash
PYTHONPATH=. uv run pytest services/shared/tests/test_observability.py -v
```

Expected:
```
test_get_logger_returns_logger PASSED
test_get_logger_emits_json PASSED
test_setup_tracing_does_not_raise_when_jaeger_unreachable PASSED
test_setup_metrics_exposes_metrics_endpoint PASSED
4 passed
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml services/shared/ uv.lock
git commit -m "feat: add shared observability module (OTel tracing, Prometheus metrics, JSON logging)"
```

---

## Task 2: Infrastructure — Jaeger, Prometheus, prometheus.yml

**Files:**
- Create: `observability/prometheus.yml`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Create `observability/prometheus.yml`**

```bash
mkdir -p observability
```

Write `observability/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: api-gateway
    static_configs:
      - targets: ["api-gateway:8000"]

  - job_name: model-gateway
    static_configs:
      - targets: ["model-gateway:8001"]

  - job_name: document-svc
    static_configs:
      - targets: ["document-svc:8002"]

  - job_name: evaluation-svc
    static_configs:
      - targets: ["evaluation-svc:8003"]

  - job_name: job-runner
    static_configs:
      - targets: ["job-runner:8004"]
```

- [ ] **Step 2: Add `jaeger` and `prometheus` services to `docker-compose.yml`**

Insert these two services at the bottom of the `services:` block, before `volumes:`:

```yaml
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "4317:4317"
    environment:
      COLLECTOR_OTLP_ENABLED: "true"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./observability/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - api-gateway
      - model-gateway
      - document-svc
      - evaluation-svc
      - job-runner
```

- [ ] **Step 3: Add `OTEL_EXPORTER_OTLP_ENDPOINT` and `LOG_LEVEL` to all 5 app services**

For each of the 5 application services (`evaluation-svc`, `model-gateway`, `document-svc`, `job-runner`, `api-gateway`), add to their `environment:` block:

```yaml
      OTEL_EXPORTER_OTLP_ENDPOINT: http://jaeger:4317
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
```

- [ ] **Step 4: Validate compose syntax**

```bash
docker compose config --quiet
```

Expected: no output (valid YAML).

- [ ] **Step 5: Commit**

```bash
git add observability/prometheus.yml docker-compose.yml
git commit -m "feat: add Jaeger and Prometheus containers, OTel env vars to all services"
```

---

## Task 3: Update all 5 Dockerfiles

**Files:**
- Modify: `services/evaluation-svc/Dockerfile`
- Modify: `services/model-gateway/Dockerfile`
- Modify: `services/document-svc/Dockerfile`
- Modify: `services/job-runner/Dockerfile`
- Modify: `services/api-gateway/Dockerfile`

Each Dockerfile needs the OTel + Prometheus packages installed and the shared module copied in. The pattern is the same for all 5.

- [ ] **Step 1: Update `services/evaluation-svc/Dockerfile`**

Replace the existing file with:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    pydantic-ai>=1.0.0 \
    pydantic>=2.0 \
    pillow>=10.0 \
    requests>=2.31.0 \
    "opentelemetry-sdk>=1.20" \
    "opentelemetry-instrumentation-fastapi>=0.41b0" \
    "opentelemetry-exporter-otlp-proto-grpc>=1.20" \
    "opentelemetry-instrumentation-httpx>=0.41b0" \
    "prometheus-fastapi-instrumentator>=6.1" \
    "prometheus-client>=0.19"

COPY src/ /app/src/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e . --no-deps

RUN mkdir -p /app/shared && touch /app/shared/__init__.py
COPY services/shared/observability.py /app/shared/observability.py
COPY services/evaluation-svc/main.py /app/main.py
COPY services/evaluation-svc/judge.py /app/judge.py
COPY services/evaluation-svc/rag_scorer.py /app/rag_scorer.py

EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
```

- [ ] **Step 2: Update `services/model-gateway/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "httpx>=0.27" \
    "opentelemetry-sdk>=1.20" \
    "opentelemetry-instrumentation-fastapi>=0.41b0" \
    "opentelemetry-exporter-otlp-proto-grpc>=1.20" \
    "opentelemetry-instrumentation-httpx>=0.41b0" \
    "prometheus-fastapi-instrumentator>=6.1" \
    "prometheus-client>=0.19"

RUN mkdir -p /app/shared && touch /app/shared/__init__.py
COPY services/shared/observability.py /app/shared/observability.py
COPY services/model-gateway/main.py /app/main.py
COPY services/model-gateway/pool.py /app/pool.py
COPY services/model-gateway/clients.py /app/clients.py

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

- [ ] **Step 3: Update `services/document-svc/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "redis[hiredis]>=5.0" \
    "redisvl>=0.2" \
    "sentence-transformers>=3.0" \
    "pillow>=10.0" \
    "opentelemetry-sdk>=1.20" \
    "opentelemetry-instrumentation-fastapi>=0.41b0" \
    "opentelemetry-exporter-otlp-proto-grpc>=1.20" \
    "opentelemetry-instrumentation-httpx>=0.41b0" \
    "prometheus-fastapi-instrumentator>=6.1" \
    "prometheus-client>=0.19"

RUN mkdir -p /app/shared && touch /app/shared/__init__.py
COPY services/shared/observability.py /app/shared/observability.py
COPY services/document-svc/main.py /app/main.py
COPY services/document-svc/indexer.py /app/indexer.py
COPY services/document-svc/search.py /app/search.py

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
```

- [ ] **Step 4: Update `services/job-runner/Dockerfile`**

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
    "pypdf>=4.0" \
    "opentelemetry-sdk>=1.20" \
    "opentelemetry-instrumentation-fastapi>=0.41b0" \
    "opentelemetry-exporter-otlp-proto-grpc>=1.20" \
    "opentelemetry-instrumentation-httpx>=0.41b0" \
    "prometheus-fastapi-instrumentator>=6.1" \
    "prometheus-client>=0.19"

RUN pip install --no-cache-dir spacy>=3.7 && \
    python -m spacy download en_core_web_sm

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e . --no-deps

RUN mkdir -p /app/shared && touch /app/shared/__init__.py
COPY services/shared/observability.py /app/shared/observability.py
COPY services/job-runner/main.py /app/main.py
COPY services/job-runner/state.py /app/state.py
COPY services/job-runner/jobs/ /app/jobs/

EXPOSE 8004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
```

- [ ] **Step 5: Update `services/api-gateway/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "httpx>=0.27" \
    "opentelemetry-sdk>=1.20" \
    "opentelemetry-instrumentation-fastapi>=0.41b0" \
    "opentelemetry-exporter-otlp-proto-grpc>=1.20" \
    "opentelemetry-instrumentation-httpx>=0.41b0" \
    "prometheus-fastapi-instrumentator>=6.1" \
    "prometheus-client>=0.19"

RUN mkdir -p /app/shared && touch /app/shared/__init__.py
COPY services/shared/observability.py /app/shared/observability.py
COPY services/api-gateway/main.py /app/main.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 6: Verify compose syntax (Dockerfiles aren't validated until build)**

```bash
docker compose config --quiet
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add services/evaluation-svc/Dockerfile \
        services/model-gateway/Dockerfile \
        services/document-svc/Dockerfile \
        services/job-runner/Dockerfile \
        services/api-gateway/Dockerfile
git commit -m "feat: add OTel + Prometheus deps and shared observability module to all service Dockerfiles"
```

---

## Task 4: Wire tracing + metrics into all 5 service main.py files

**Files:**
- Modify: `services/api-gateway/main.py`
- Modify: `services/evaluation-svc/main.py`
- Modify: `services/document-svc/main.py`
- Modify: `services/model-gateway/main.py`
- Modify: `services/job-runner/main.py`

Each service gets the same three-line addition right after `app = FastAPI(...)`. Existing tests still pass because `setup_tracing` and `setup_metrics` are no-ops when called without a running Jaeger/Prometheus scraper.

- [ ] **Step 1: Update `services/api-gateway/main.py`**

After the `app = FastAPI(...)` line, add:

```python
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from shared.observability import setup_tracing, setup_metrics, get_logger
setup_tracing("api-gateway")
setup_metrics(app)
logger = get_logger("api-gateway")
```

The full updated `services/api-gateway/main.py`:

```python
import os
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="api-gateway", version="1.0")

from shared.observability import setup_tracing, setup_metrics, get_logger
setup_tracing("api-gateway")
setup_metrics(app)
logger = get_logger("api-gateway")

_JOB_RUNNER_URL = os.getenv("JOB_RUNNER_URL", "http://job-runner:8004")


class JobRequest(BaseModel):
    type: Literal["corrupt", "benchmark", "mitigation", "index"]
    dataset: Optional[str] = None
    config: dict = {}


@app.post("/jobs")
def submit_job(req: JobRequest):
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

- [ ] **Step 2: Update `services/evaluation-svc/main.py`**

Add after `app = FastAPI(...)`:

```python
from shared.observability import setup_tracing, setup_metrics, get_logger
setup_tracing("evaluation-svc")
setup_metrics(app)
logger = get_logger("evaluation-svc")
```

The full updated file (only the top portion changes; routes stay identical):

```python
import sys, os
sys.path.insert(0, "/app")

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

import judge as judge_module
import rag_scorer

app = FastAPI(title="evaluation-svc", version="1.0")

from shared.observability import setup_tracing, setup_metrics, get_logger
setup_tracing("evaluation-svc")
setup_metrics(app)
logger = get_logger("evaluation-svc")


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
        logger.warning("Judge call failed", extra={"error": str(e)})
        return AnswerabilityResponse(verdict=None, confidence=0.0, reason=str(e))


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
    try:
        result = rag_scorer.score_rag(
            req.question, req.retrieved_context, req.model_answer, req.ground_truth
        )
        return RAGEvalResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MetricsRequest(BaseModel):
    y_true: list[bool]
    y_pred: list[bool]

    @model_validator(mode="after")
    def lengths_must_match(self):
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                f"y_true length ({len(self.y_true)}) must equal y_pred length ({len(self.y_pred)})"
            )
        return self


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
    try:
        from src.benchmark.evaluation.metrics import compute_metrics
        m = compute_metrics(req.y_true, req.y_pred)
        return MetricsResponse(**m.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 3: Update `services/document-svc/main.py`**

Add after `app = FastAPI(...)`:

```python
from shared.observability import setup_tracing, setup_metrics, get_logger
setup_tracing("document-svc")
setup_metrics(app)
logger = get_logger("document-svc")
```

Full updated file:

```python
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis as sync_redis

import indexer
import search as search_module

app = FastAPI(title="document-svc", version="1.0")

from shared.observability import setup_tracing, setup_metrics, get_logger
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
```

- [ ] **Step 4: Run existing tests — verify nothing broken**

```bash
PYTHONPATH=. uv run pytest services/api-gateway/tests/ services/evaluation-svc/tests/ services/document-svc/tests/ -v 2>&1 | tail -10
```

Expected: all previously passing tests still pass (15 total).

- [ ] **Step 5: Commit**

```bash
git add services/api-gateway/main.py \
        services/evaluation-svc/main.py \
        services/document-svc/main.py
git commit -m "feat: wire OTel tracing + Prometheus metrics into api-gateway, evaluation-svc, document-svc"
```

---

## Task 5: Custom metrics in model-gateway

**Files:**
- Modify: `services/model-gateway/clients.py`
- Modify: `services/model-gateway/main.py`
- Test: `services/model-gateway/tests/test_model_gateway.py`

- [ ] **Step 1: Write failing tests for new metrics behavior**

Append to `services/model-gateway/tests/test_model_gateway.py`:

```python
def test_infer_latency_metric_registered():
    """_INFER_LATENCY histogram must be registered in prometheus_client REGISTRY."""
    from prometheus_client import REGISTRY
    metric_names = [m.name for m in REGISTRY.collect()]
    assert "inference_latency_seconds" in metric_names


def test_infer_error_metric_registered():
    from prometheus_client import REGISTRY
    metric_names = [m.name for m in REGISTRY.collect()]
    assert "inference_errors_total" in metric_names
```

- [ ] **Step 2: Run — verify new tests fail**

```bash
PYTHONPATH=. uv run pytest services/model-gateway/tests/test_model_gateway.py::test_infer_latency_metric_registered -v
```

Expected: `FAILED — AssertionError: assert "inference_latency_seconds" in [...]`

- [ ] **Step 3: Add metrics to `services/model-gateway/clients.py`**

At the top of the file, after the existing imports, add:

```python
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
```

Wrap the dispatch block in `async_infer` to record latency and errors. Replace the body of `async_infer` with:

```python
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
```

- [ ] **Step 4: Add `_POOL_HEALTHY` gauge + lifespan to `services/model-gateway/main.py`**

Replace the entire file with:

```python
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from prometheus_client import Gauge
from pydantic import BaseModel

from pool import WorkerPool
import clients

_POOL_HEALTHY = Gauge(
    "pool_healthy_workers",
    "Number of healthy workers in each local pool",
    labelnames=["pool"],
)

_LLAMA_URLS_RAW = os.getenv("LLAMA_URLS", "")
_llama_pool = WorkerPool([u.strip() for u in _LLAMA_URLS_RAW.split(",") if u.strip()])

_VLLM_URLS_RAW = os.getenv("VLLM_URLS", "")
_vllm_pool = WorkerPool([u.strip() for u in _VLLM_URLS_RAW.split(",") if u.strip()])

_ENABLED = set(os.getenv("ENABLED_MODELS", "llama,vllm").split(","))
_ALL_MODELS = ["llama", "vllm", "mistral", "google", "openrouter"]


async def _poll_pool_health() -> None:
    """Background task: update pool health gauges every 30 s."""
    while True:
        _POOL_HEALTHY.labels(pool="llama").set(
            sum(1 for w in _llama_pool.status() if w["healthy"])
        )
        _POOL_HEALTHY.labels(pool="vllm").set(
            sum(1 for w in _vllm_pool.status() if w["healthy"])
        )
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_poll_pool_health())
    yield
    task.cancel()


app = FastAPI(title="model-gateway", version="1.0", lifespan=lifespan)

from shared.observability import setup_tracing, setup_metrics, get_logger
setup_tracing("model-gateway")
setup_metrics(app)
logger = get_logger("model-gateway")


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

- [ ] **Step 5: Run all model-gateway tests**

```bash
PYTHONPATH=. uv run pytest services/model-gateway/tests/ -v
```

Expected: `9 passed` (7 existing + 2 new metric registration tests).

- [ ] **Step 6: Commit**

```bash
git add services/model-gateway/clients.py services/model-gateway/main.py \
        services/model-gateway/tests/test_model_gateway.py
git commit -m "feat: add inference latency/error metrics and pool health gauge to model-gateway"
```

---

## Task 6: Job counter + job_id trace in job-runner

**Files:**
- Modify: `services/job-runner/state.py`
- Modify: `services/job-runner/main.py`
- Modify: `services/job-runner/tests/test_job_runner.py`

- [ ] **Step 1: Write failing tests**

Append to `services/job-runner/tests/test_job_runner.py`:

```python
def test_jobs_counter_registered():
    """jobs_total counter must be registered in prometheus REGISTRY."""
    from prometheus_client import REGISTRY
    names = [m.name for m in REGISTRY.collect()]
    assert "jobs_total" in names


def test_jobs_counter_increments_on_done(r):
    from prometheus_client import REGISTRY

    job_id = state.create_job(r, "benchmark")
    # snapshot before
    before = {
        s.labels: s.value
        for m in REGISTRY.collect() if m.name == "jobs_total"
        for s in m.samples
    }

    state.update_job(r, job_id, status="done")

    after = {
        s.labels: s.value
        for m in REGISTRY.collect() if m.name == "jobs_total"
        for s in m.samples
    }

    key = {"job_type": "benchmark", "status": "done"}
    assert after.get(key, 0) > before.get(key, 0)
```

- [ ] **Step 2: Run — verify new tests fail**

```bash
PYTHONPATH=. uv run pytest services/job-runner/tests/test_job_runner.py::test_jobs_counter_registered -v
```

Expected: `FAILED — AssertionError: assert "jobs_total" in [...]`

- [ ] **Step 3: Update `services/job-runner/state.py`**

Replace the entire file with:

```python
import uuid
from datetime import datetime, timezone
from typing import Optional

from prometheus_client import Counter

JOB_TTL_SECONDS = 86400  # 24 hours

_JOBS = Counter(
    "jobs_total",
    "Job status transitions to terminal states",
    labelnames=["job_type", "status"],
)

_TERMINAL_STATUSES = {"done", "failed", "cancelled"}


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
    status = str(fields.get("status", ""))
    if status in _TERMINAL_STATUSES:
        job = get_job(redis, job_id)
        job_type = job.get("type", "unknown") if job else "unknown"
        _JOBS.labels(job_type=job_type, status=status).inc()


def get_job(redis, job_id: str) -> Optional[dict]:
    """Return the job dict or None if not found."""
    data = redis.hgetall(f"job:{job_id}")
    if not data:
        return None
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

- [ ] **Step 4: Update `services/job-runner/main.py` — add setup_tracing + job_id span**

Replace the entire file with:

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

from shared.observability import setup_tracing, setup_metrics, get_logger
setup_tracing("job-runner")
setup_metrics(app)
logger = get_logger("job-runner")

_REDIS_URL = os.getenv("REDIS_URL", "redis://redis-stack:6379")
_JOB_HANDLERS = {
    "corrupt": run_corrupt_job,
    "benchmark": run_benchmark_job,
    "mitigation": run_mitigation_job,
    "index": run_index_job,
}


def _get_redis() -> sync_redis.Redis:
    return sync_redis.from_url(_REDIS_URL, decode_responses=True)


class DispatchRequest(BaseModel):
    type: str
    config: dict


@app.post("/jobs/dispatch")
async def dispatch_job(req: DispatchRequest, background_tasks: BackgroundTasks):
    if req.type not in _JOB_HANDLERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown job type: {req.type}. Valid: {list(_JOB_HANDLERS)}",
        )

    r = _get_redis()
    job_id = state.create_job(r, req.type)
    handler = _JOB_HANDLERS[req.type]
    logger.info("Job dispatched", extra={"job_id": job_id, "job_type": req.type})

    async def _run():
        from opentelemetry import trace
        tracer = trace.get_tracer("job-runner")
        with tracer.start_as_current_span(
            f"job.{req.type}",
            attributes={"job.id": job_id, "job.type": req.type},
        ):
            r2 = _get_redis()
            try:
                await handler(r2, job_id, req.config)
            except Exception as exc:
                current = state.get_job(r2, job_id)
                if current and current.get("status") not in ("done", "failed", "cancelled"):
                    state.update_job(r2, job_id, status="failed", error=str(exc))
                logger.error("Job failed", extra={"job_id": job_id, "error": str(exc)})

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
    logger.info("Job cancelled", extra={"job_id": job_id})
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/jobs/{job_id}/logs")
def stream_logs(job_id: str):
    r = _get_redis()

    def event_generator():
        import time, json
        while True:
            data = state.get_job(r, job_id)
            if data is None:
                yield f"data: job not found\n\n"
                return
            yield f"data: {json.dumps(data)}\n\n"
            if data.get("status") in ("done", "failed", "cancelled"):
                return
            time.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Run all job-runner tests**

```bash
PYTHONPATH=. uv run pytest services/job-runner/tests/ -v
```

Expected: `10 passed` (8 existing + 2 new counter tests).

- [ ] **Step 6: Commit**

```bash
git add services/job-runner/state.py services/job-runner/main.py \
        services/job-runner/tests/test_job_runner.py
git commit -m "feat: add jobs_total counter and job_id OTel span to job-runner"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `shared/observability.py` with `setup_tracing`, `setup_metrics`, `get_logger` | Task 1 |
| `setup_tracing` safe when Jaeger unreachable | Task 1 (try/except) |
| `setup_metrics` exposes `/metrics` endpoint | Task 1 (test verifies) |
| JSON structured logging with extra fields | Task 1 |
| `opentelemetry-*` + `prometheus-*` deps in `services` extra | Task 1 |
| `observability/prometheus.yml` with all 5 scrape targets | Task 2 |
| `jaeger` + `prometheus` containers in `docker-compose.yml` | Task 2 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` + `LOG_LEVEL` on all app services | Task 2 |
| OTel + Prometheus pip installs in all 5 Dockerfiles | Task 3 |
| Shared module copied into all 5 containers | Task 3 |
| `setup_tracing` + `setup_metrics` in all 5 `main.py` | Task 4 + 5 + 6 |
| `inference_latency_seconds` histogram | Task 5 (clients.py) |
| `inference_errors_total` counter | Task 5 (clients.py) |
| `pool_healthy_workers` gauge (llama + vllm) | Task 5 (main.py lifespan) |
| `jobs_total` counter on terminal status | Task 6 (state.py) |
| `job_id` OTel span attribute on background tasks | Task 6 (main.py) |
| Structured log calls replacing bare prints | Tasks 4, 5, 6 |

All spec requirements covered. No gaps.

**Type consistency:** `setup_tracing(service_name: str)` called identically in Tasks 4, 5, 6. `_POOL_HEALTHY.labels(pool="llama")` consistent between definition (Task 5) and the polling loop (Task 5 same file). `_JOBS.labels(job_type=..., status=...)` defined and incremented in same file (Task 6 state.py).
