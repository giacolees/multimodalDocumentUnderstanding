"""Mitigation job: runs prompt strategies and RAG via document-svc + model-gateway."""

import asyncio
import json
import os
from pathlib import Path

import httpx

import state


MODEL_GATEWAY_URL = os.getenv("MODEL_GATEWAY_URL", "http://model-gateway:8001")
DOCUMENT_SVC_URL = os.getenv("DOCUMENT_SVC_URL", "http://document-svc:8002")
EVALUATION_SVC_URL = os.getenv("EVALUATION_SVC_URL", "http://evaluation-svc:8003")


def _build_prompt(strategy: str, question: str, context_chunks: list[dict]) -> str:
    import sys
    sys.path.insert(0, "/app")

    if strategy == "few_shot":
        from src.mitigation.strategies.few_shot import build_few_shot_prompt
        return build_few_shot_prompt(question)

    if strategy == "chain_of_thought":
        from src.mitigation.strategies.chain_of_thought import build_cot_prompt
        return build_cot_prompt(question)

    if strategy in ("knowledge_injection", "rag"):
        from src.mitigation.strategies.knowledge_injection import (
            DocumentMetadata, build_knowledge_injection_prompt
        )
        if context_chunks:
            entities = {"Retrieved context": [c["text"] for c in context_chunks]}
            metadata = DocumentMetadata(entities=entities)
        else:
            metadata = DocumentMetadata()
        return build_knowledge_injection_prompt(question, metadata)

    raise ValueError(f"Unknown strategy: {strategy}")


async def _run_strategy(
    client: httpx.AsyncClient,
    item: dict,
    strategy: str,
    model_id: str | None,
) -> dict:
    """Run a single strategy for a single item. Returns a result dict."""
    context_chunks = []
    if strategy == "rag":
        try:
            search_resp = await client.post(
                f"{DOCUMENT_SVC_URL}/search",
                json={"query": item["corrupted_question"], "top_k": 5, "alpha": 0.5},
            )
            if search_resp.status_code == 200:
                context_chunks = search_resp.json().get("chunks", [])
        except Exception:
            pass

    prompt = _build_prompt(strategy, item["corrupted_question"], context_chunks)
    infer_payload = {
        "document_path": item["document_path"],
        "prompt": prompt,
        "max_tokens": 256,
    }
    if model_id:
        infer_payload["model_id"] = model_id

    infer_resp = await client.post(f"{MODEL_GATEWAY_URL}/infer", json=infer_payload)
    infer_data = infer_resp.json() if infer_resp.status_code == 200 else {}

    eval_data = {}
    if strategy == "rag" and infer_data:
        eval_resp = await client.post(
            f"{EVALUATION_SVC_URL}/evaluate/rag",
            json={
                "question": item["corrupted_question"],
                "retrieved_context": [c["text"] for c in context_chunks],
                "model_answer": infer_data.get("raw_response", ""),
                "ground_truth": "unanswerable",
            },
        )
        if eval_resp.status_code == 200:
            eval_data = eval_resp.json()

    return {
        "sample_id": item["sample_id"],
        "strategy": strategy,
        "predicted_unanswerable": infer_data.get("predicted_unanswerable"),
        "raw_response": infer_data.get("raw_response", ""),
        "rag_score": eval_data.get("score"),
        "rag_correct": eval_data.get("correct"),
    }


async def run_mitigation_job(redis, job_id: str, config: dict) -> None:
    corrupted_path = config["corrupted_dataset"]
    output_dir = config.get("output_dir", "data/results/mitigation")
    strategies = config.get("strategies", ["few_shot", "chain_of_thought", "knowledge_injection", "rag"])
    model_id = config.get("model_id")

    with open(corrupted_path) as f:
        dataset: list[dict] = json.load(f)

    total = len(dataset) * len(strategies)
    state.update_job(redis, job_id, status="running", total=total)

    results: dict[str, list] = {s: [] for s in strategies}
    completed = 0

    async with httpx.AsyncClient(timeout=180.0) as client:
        for item in dataset:
            # Run all strategies for this sample concurrently
            strategy_tasks = [
                _run_strategy(client, item, strategy, model_id)
                for strategy in strategies
            ]
            strategy_results = await asyncio.gather(*strategy_tasks, return_exceptions=True)
            for strategy, result in zip(strategies, strategy_results):
                if not isinstance(result, Exception):
                    results[strategy].append(result)
            completed += len(strategies)
            state.update_job(redis, job_id, progress=completed)

    out_path = Path(output_dir) / f"{job_id}_mitigation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    state.update_job(redis, job_id, status="done", result_path=str(out_path))
