# Observability Design
**Date:** 2026-06-06
**Status:** Approved

## Context

The microservices stack (api-gateway, model-gateway, document-svc, evaluation-svc, job-runner) currently has no tracing, no metrics, and no structured logging. Failures and latency spikes are invisible — you only discover problems when a job times out or returns wrong results.

## Goals

- Distributed traces across all 5 FastAPI services via OpenTelemetry → Jaeger
- Prometheus metrics: inference latency, pool health, job counters, error rates
- Structured JSON logs with `job_id` propagation (stdout only, no log aggregation service)
- Minimal operational overhead: 2 new containers (Jaeger, Prometheus), one shared module

## Non-goals

- Grafana dashboards (Prometheus UI is sufficient for now)
- Log aggregation (Loki/ELK)
- Alerting rules
- Instrumenting llama.cpp / vLLM servers (they're black boxes; latency captured at the client call in `clients.py`)

---

## Architecture

```
┌─────────────┐    OTLP gRPC     ┌──────────────┐
│ api-gateway │ ────────────────► │              │
│ model-gw    │                  │  Jaeger       │ :16686 (UI)
│ document-svc│ ────────────────► │  all-in-one  │ :4317  (OTLP)
│ eval-svc    │                  │              │
│ job-runner  │ ────────────────► └──────────────┘

┌─────────────┐    GET /metrics  ┌──────────────┐
│ api-gateway │ ◄────────────── │              │
│ model-gw    │                  │  Prometheus  │ :9090
│ document-svc│ ◄────────────── │  (scrapes    │
│ eval-svc    │                  │   every 15s) │
│ job-runner  │ ◄────────────── └──────────────┘
└─────────────┘

httpx outbound calls (model-gateway → workers, job-runner → services)
are automatically captured as child spans via opentelemetry-instrumentation-httpx.
```

---

## New Files

| File | Purpose |
|---|---|
| `services/shared/observability.py` | `setup_tracing(service_name)`, `setup_metrics(app)`, `get_logger(name)` |
| `observability/prometheus.yml` | Prometheus scrape config for all 5 services |

## Modified Files

| File | Change |
|---|---|
| `docker-compose.yml` | Add `jaeger`, `prometheus` services; add `OTEL_EXPORTER_OTLP_ENDPOINT` to each app service |
| `pyproject.toml` | Add OTel + Prometheus deps to `services` extra |
| `services/*/main.py` (all 5) | Call `setup_tracing` + `setup_metrics` at startup |
| `services/*/Dockerfile` (all 5) | `COPY services/shared/observability.py /app/shared/observability.py` |
| `services/model-gateway/clients.py` | Emit `inference_latency_seconds` histogram + `inference_errors_total` counter |
| `services/model-gateway/main.py` | Emit `pool_healthy_workers` gauge on startup |
| `services/job-runner/state.py` | Emit `jobs_total` counter on status transitions |

---

## `services/shared/observability.py`

```python
"""Shared observability setup: tracing (OTel → Jaeger) + metrics (Prometheus)."""

import logging
import os
from typing import Optional


def setup_tracing(service_name: str) -> None:
    """Configure OTel SDK to export traces to Jaeger via OTLP gRPC."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import Resource

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()


def setup_metrics(app) -> None:
    """Attach prometheus-fastapi-instrumentator to the FastAPI app."""
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def get_logger(name: str) -> logging.Logger:
    """Return a logger that emits JSON-formatted records to stdout."""
    import json

    class _JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            # Merge any extra fields attached to the record
            for key, val in record.__dict__.items():
                if key not in (
                    "args", "asctime", "created", "exc_info", "exc_text",
                    "filename", "funcName", "levelname", "levelno", "lineno",
                    "message", "module", "msecs", "msg", "name", "pathname",
                    "process", "processName", "relativeCreated", "stack_info",
                    "thread", "threadName",
                ):
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

`get_logger` adds any extra kwargs passed via `logger.info("msg", extra={"job_id": "abc"})` directly into the JSON record — no special setup required at call sites.

---

## Per-Service Changes (`main.py`)

Each of the 5 services gets these lines added at startup — before the route definitions, after `app = FastAPI(...)`:

```python
from shared.observability import setup_tracing, setup_metrics, get_logger

setup_tracing("model-gateway")   # service name differs per file
setup_metrics(app)
logger = get_logger("model-gateway")
```

Replace any existing `print()` calls with `logger.info(...)`.

---

## Custom Metrics in `model-gateway/clients.py`

```python
from prometheus_client import Counter, Histogram

_INFER_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference round-trip latency",
    labelnames=["model_id"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 180],
)
_INFER_ERRORS = Counter(
    "inference_errors_total",
    "Inference failures by model and error type",
    labelnames=["model_id", "error_type"],
)
```

In `async_infer`, wrap the dispatch:

```python
with _INFER_LATENCY.labels(model_id=model_id).time():
    try:
        result = await _dispatch(...)
    except Exception as e:
        _INFER_ERRORS.labels(model_id=model_id, error_type=type(e).__name__).inc()
        raise
```

---

## Pool Health Gauge in `model-gateway/main.py`

```python
from prometheus_client import Gauge

_POOL_HEALTHY = Gauge(
    "pool_healthy_workers",
    "Number of healthy workers in each local pool",
    labelnames=["pool"],
)
```

Updated via a FastAPI lifespan background task that calls `GET /models/{id}/health` every 30 seconds and sets:

```python
_POOL_HEALTHY.labels(pool="llama").set(
    sum(1 for w in _llama_pool.status() if w["healthy"])
)
_POOL_HEALTHY.labels(pool="vllm").set(
    sum(1 for w in _vllm_pool.status() if w["healthy"])
)
```

---

## Job Counter in `job-runner/state.py`

```python
from prometheus_client import Counter

_JOBS = Counter(
    "jobs_total",
    "Job status transitions",
    labelnames=["job_type", "status"],
)
```

Called inside `update_job` when `status` field changes to a terminal value:

```python
if "status" in fields and fields["status"] in ("done", "failed", "cancelled"):
    _JOBS.labels(job_type=get_job(redis, job_id).get("type", "unknown"),
                 status=fields["status"]).inc()
```

---

## `observability/prometheus.yml`

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

---

## docker-compose.yml additions

### New services

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

### Per application service — add env var

```yaml
environment:
  OTEL_EXPORTER_OTLP_ENDPOINT: http://jaeger:4317
  LOG_LEVEL: ${LOG_LEVEL:-INFO}
```

---

## Dockerfile changes (all 5 services)

Add one line before `CMD`:

```dockerfile
COPY services/shared/observability.py /app/shared/__init__.py
COPY services/shared/observability.py /app/shared/observability.py
```

Wait — `shared/` needs an `__init__.py`. Add it as an empty file:

```dockerfile
RUN mkdir -p /app/shared && touch /app/shared/__init__.py
COPY services/shared/observability.py /app/shared/observability.py
```

---

## pyproject.toml — services extra additions

```toml
"opentelemetry-sdk>=1.20",
"opentelemetry-instrumentation-fastapi>=0.41b0",
"opentelemetry-exporter-otlp-proto-grpc>=1.20",
"opentelemetry-instrumentation-httpx>=0.41b0",
"prometheus-fastapi-instrumentator>=6.1",
"prometheus-client>=0.19",
```

---

## `job_id` Propagation in Traces

In `job-runner/main.py`, when a background task starts, attach the `job_id` as an OTel span attribute:

```python
from opentelemetry import trace

async def _run():
    with trace.get_tracer(__name__).start_as_current_span(
        f"job.{req.type}", attributes={"job.id": job_id, "job.type": req.type}
    ):
        await handler(r2, job_id, req.config)
```

This links the background task's spans to the original HTTP request span, making the full job lifecycle visible as one trace in Jaeger.

---

## Error Handling

- OTel SDK is initialized at startup; if Jaeger is unreachable, `BatchSpanProcessor` drops spans silently (no crash). Services continue functioning without tracing.
- If Prometheus scrape fails, Prometheus retries on next interval. Services are unaffected.
- `setup_tracing` and `setup_metrics` wrapped in try/except in production: if OTel deps are missing (e.g. Dockerfile not updated), the service starts with a warning log, not a crash.

---

## Testing

- Unit tests: no changes needed — OTel and Prometheus are initialized at module level but don't break existing mocks.
- Integration: after `docker compose up -d`, `curl http://localhost:8001/metrics` should return Prometheus text format. Jaeger UI at `http://localhost:16686` should show traces after any `/infer` call.
