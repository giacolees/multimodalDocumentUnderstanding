import sys, os, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Load main.py under a unique module name to avoid sys.modules['main'] collisions
# when all three service test suites run in the same pytest process.
_spec = importlib.util.spec_from_file_location(
    "api_gateway_main",
    os.path.join(os.path.dirname(__file__), "..", "main.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["api_gateway_main"] = _mod
_spec.loader.exec_module(_mod)
app = _mod.app

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
