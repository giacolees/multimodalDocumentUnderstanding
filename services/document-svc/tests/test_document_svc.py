import sys, os, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Load main.py under a unique module name to avoid sys.modules['main'] collisions
# when all three service test suites run in the same pytest process.
_spec = importlib.util.spec_from_file_location(
    "document_svc_main",
    os.path.join(os.path.dirname(__file__), "..", "main.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["document_svc_main"] = _mod
_spec.loader.exec_module(_mod)
app = _mod.app

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
        resp = client.post("/search", json={"query": "anything", "top_k": 3})
    assert resp.status_code == 200
    call_kwargs = mock_search.call_args.kwargs
    assert call_kwargs["alpha"] == 0.5
    assert call_kwargs["top_k"] == 3
    assert call_kwargs["query"] == "anything"


def test_index_triggers_indexer():
    with patch("indexer.index_dataset", return_value={"chunks_indexed": 42}) as mock_idx:
        resp = client.post("/documents/index", json={
            "dataset": "docvqa",
            "data_dir": "data/raw/docvqa",
        })
    assert resp.status_code == 200
    assert resp.json()["chunks_indexed"] == 42
