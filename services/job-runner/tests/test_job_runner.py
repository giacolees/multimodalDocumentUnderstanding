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
