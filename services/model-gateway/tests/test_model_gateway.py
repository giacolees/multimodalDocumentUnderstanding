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


def test_infer_unknown_model_returns_400():
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
