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
