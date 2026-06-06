import os
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from shared.observability import setup_tracing, setup_metrics, get_logger

app = FastAPI(title="api-gateway", version="1.0")
setup_tracing("api-gateway")
setup_metrics(app)
logger = get_logger("api-gateway")

_JOB_RUNNER_URL = os.getenv("JOB_RUNNER_URL", "http://job-runner:8004")


class JobRequest(BaseModel):
    type: Literal["corrupt", "benchmark", "mitigation", "index"]
    dataset: Optional[str] = None
    config: dict = {}


@app.post("/jobs")
def submit_job(req: JobRequest):
    dispatch_payload = {"type": req.type, "config": req.config}
    if req.dataset:
        dispatch_payload["config"].setdefault("dataset", req.dataset)

    logger.info("Submitting job", extra={"job_type": req.type, "dataset": req.dataset})
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{_JOB_RUNNER_URL}/jobs/dispatch", json=dispatch_payload)
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        logger.error("job-runner unavailable")
        raise HTTPException(status_code=503, detail="job-runner unavailable")


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{_JOB_RUNNER_URL}/jobs/{job_id}")
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="job-runner unavailable")


@app.get("/jobs")
def list_jobs(offset: int = 0, limit: int = 50):
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{_JOB_RUNNER_URL}/jobs", params={"offset": offset, "limit": limit})
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="job-runner unavailable")


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.delete(f"{_JOB_RUNNER_URL}/jobs/{job_id}")
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="job-runner unavailable")


@app.get("/health")
def health():
    return {"status": "ok"}
