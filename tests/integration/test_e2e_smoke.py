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
