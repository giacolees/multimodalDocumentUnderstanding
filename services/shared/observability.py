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
