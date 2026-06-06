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
    setup_tracing("test-service")


def test_setup_metrics_exposes_metrics_endpoint():
    from shared.observability import setup_metrics
    app = FastAPI()
    setup_metrics(app)
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "python_info" in resp.text
