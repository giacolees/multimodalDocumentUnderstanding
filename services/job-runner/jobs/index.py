"""Index job: triggers document-svc to chunk and embed a dataset."""

import os
import httpx
import state


DOCUMENT_SVC_URL = os.getenv("DOCUMENT_SVC_URL", "http://document-svc:8002")


async def run_index_job(redis, job_id: str, config: dict) -> None:
    state.update_job(redis, job_id, status="running")
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            resp = await client.post(
                f"{DOCUMENT_SVC_URL}/documents/index",
                json={"dataset": config["dataset"], "data_dir": config["data_dir"]},
            )
            resp.raise_for_status()
            result = resp.json()
        state.update_job(
            redis, job_id,
            status="done",
            total=result.get("chunks_indexed", 0),
            progress=result.get("chunks_indexed", 0),
        )
    except Exception as e:
        state.update_job(redis, job_id, status="failed", error=str(e))
        raise
