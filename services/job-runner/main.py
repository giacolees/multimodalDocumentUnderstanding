import asyncio
import os
from typing import Optional

import redis as sync_redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import state
from shared.observability import setup_tracing, setup_metrics, get_logger
from jobs.corrupt import run_corrupt_job
from jobs.benchmark import run_benchmark_job
from jobs.mitigation import run_mitigation_job
from jobs.index import run_index_job

app = FastAPI(title="job-runner", version="1.0")
setup_tracing("job-runner")
setup_metrics(app)
logger = get_logger("job-runner")

_REDIS_URL = os.getenv("REDIS_URL", "redis://redis-stack:6379")
_JOB_HANDLERS = {
    "corrupt": run_corrupt_job,
    "benchmark": run_benchmark_job,
    "mitigation": run_mitigation_job,
    "index": run_index_job,
}


def _get_redis() -> sync_redis.Redis:
    return sync_redis.from_url(_REDIS_URL, decode_responses=True)


class DispatchRequest(BaseModel):
    type: str
    config: dict


@app.post("/jobs/dispatch")
async def dispatch_job(req: DispatchRequest, background_tasks: BackgroundTasks):
    if req.type not in _JOB_HANDLERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown job type: {req.type}. Valid: {list(_JOB_HANDLERS)}",
        )

    r = _get_redis()
    job_id = state.create_job(r, req.type)
    handler = _JOB_HANDLERS[req.type]
    logger.info("Job dispatched", extra={"job_id": job_id, "job_type": req.type})

    async def _run():
        from opentelemetry import trace
        tracer = trace.get_tracer("job-runner")
        with tracer.start_as_current_span(
            f"job.{req.type}",
            attributes={"job.id": job_id, "job.type": req.type},
        ):
            r2 = _get_redis()
            try:
                await handler(r2, job_id, req.config)
            except Exception as exc:
                current = state.get_job(r2, job_id)
                if current and current.get("status") not in ("done", "failed", "cancelled"):
                    state.update_job(r2, job_id, status="failed", error=str(exc))
                logger.error("Job failed", extra={"job_id": job_id, "error": str(exc)})

    background_tasks.add_task(_run)
    job_data = state.get_job(r, job_id)
    return job_data


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    r = _get_redis()
    data = state.get_job(r, job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return data


@app.get("/jobs")
def list_jobs_endpoint(offset: int = 0, limit: int = 50):
    r = _get_redis()
    return state.list_jobs(r, offset=offset, limit=limit)


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    r = _get_redis()
    data = state.get_job(r, job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    state.update_job(r, job_id, status="cancelled")
    logger.info("Job cancelled", extra={"job_id": job_id})
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/jobs/{job_id}/logs")
def stream_logs(job_id: str):
    r = _get_redis()

    def event_generator():
        import time, json
        while True:
            data = state.get_job(r, job_id)
            if data is None:
                yield f"data: job not found\n\n"
                return
            yield f"data: {json.dumps(data)}\n\n"
            if data.get("status") in ("done", "failed", "cancelled"):
                return
            time.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}
