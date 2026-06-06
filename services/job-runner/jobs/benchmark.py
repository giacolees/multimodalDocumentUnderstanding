"""Benchmark job: fans out to all configured models via model-gateway concurrently."""

import asyncio
import json
import os
from pathlib import Path

import httpx

import state


MODEL_GATEWAY_URL = os.getenv("MODEL_GATEWAY_URL", "http://model-gateway:8001")

_BASELINE_PROMPT = (
    "Look at the document image and answer the following question.\n"
    "If the question cannot be answered from the document, respond with exactly: UNANSWERABLE\n"
    "Otherwise provide the answer.\n\nQuestion: {question}"
)


async def _infer_all_models(
    client: httpx.AsyncClient,
    item: dict,
    model_ids: list[str],
    prompt_template: str,
) -> dict:
    prompt = prompt_template.format(question=item["corrupted_question"])
    payload = {
        "document_path": item["document_path"],
        "prompt": prompt,
        "max_tokens": 256,
    }
    if len(model_ids) == 1:
        payload["model_id"] = model_ids[0]

    resp = await client.post(f"{MODEL_GATEWAY_URL}/infer", json=payload, timeout=180.0)
    resp.raise_for_status()
    results = resp.json()
    if isinstance(results, dict):
        results = [results]
    return {"sample_id": item["sample_id"], "results": results}


async def run_benchmark_job(redis, job_id: str, config: dict) -> None:
    corrupted_path = config["corrupted_dataset"]
    output_dir = config.get("output_dir", "data/results/benchmark")
    model_ids = config.get("model_ids", [])
    prompt_template = config.get("prompt_template", _BASELINE_PROMPT)

    with open(corrupted_path) as f:
        dataset: list[dict] = json.load(f)

    state.update_job(redis, job_id, status="running", total=len(dataset))

    aggregated: dict[str, list] = {}
    completed = 0

    async with httpx.AsyncClient() as client:
        tasks = [
            _infer_all_models(client, item, model_ids, prompt_template)
            for item in dataset
        ]
        chunk_size = 20
        for i in range(0, len(tasks), chunk_size):
            chunk_results = await asyncio.gather(
                *tasks[i: i + chunk_size], return_exceptions=True
            )
            for res in chunk_results:
                if isinstance(res, Exception):
                    continue
                for model_result in res["results"]:
                    mid = model_result.get("model_id", "unknown")
                    aggregated.setdefault(mid, []).append({
                        "sample_id": res["sample_id"],
                        "predicted_unanswerable": model_result.get("predicted_unanswerable"),
                        "raw_response": model_result.get("raw_response", ""),
                        "latency_ms": model_result.get("latency_ms", -1),
                    })
            completed += len(chunk_results)
            state.update_job(redis, job_id, progress=completed)

    out_path = Path(output_dir) / f"{job_id}_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    state.update_job(redis, job_id, status="done", result_path=str(out_path))
