import uuid
from datetime import datetime, timezone
from typing import Optional

from prometheus_client import Counter

JOB_TTL_SECONDS = 86400  # 24 hours

try:
    _JOBS = Counter(
        "jobs_total",
        "Job status transitions to terminal states",
        labelnames=["job_type", "status"],
    )
except ValueError:
    from prometheus_client import REGISTRY
    _JOBS = REGISTRY._names_to_collectors.get("jobs_total") or \
            next(c for c in REGISTRY._names_to_collectors.values()
                 if getattr(c, "_name", None) in ("jobs_total", "jobs"))

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
