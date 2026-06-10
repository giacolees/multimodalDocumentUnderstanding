"""Main corruption pipeline for Part 1.

Orchestrated with LangChain LCEL: each processing phase is a RunnableLambda
and samples flow through the composed chain via .invoke().

Usage:
    python -m src.dataset.pipeline \
        --dataset docvqa \
        --data_dir data/raw/docvqa \
        --output_dir data/corrupted \
        --config configs/dataset_config.yaml
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)
from langchain_core.runnables import RunnableLambda

from .loaders.docvqa_loader import DocVQALoader
from .loaders.dude_loader import DUDELoader
from .loaders.mp_docvqa_loader import MPDocVQALoader
from .corruption.nlp_entity_corruptor import NLPEntityCorruptor
from .corruption.element_corruptor import ElementCorruptor
from .corruption.layout_corruptor import LayoutCorruptor
from .quality_check.llm_judge import LLMJudge, JudgeResult


LOADERS = {"docvqa": DocVQALoader, "dude": DUDELoader, "mp_docvqa": MPDocVQALoader}
CORRUPTORS = [NLPEntityCorruptor, ElementCorruptor, LayoutCorruptor]

# ---------------------------------------------------------------------------
# Pipeline state type (plain dict flowing through the LCEL chain)
# Keys added progressively by each step:
#   input:  sample, corruptors, judge (or None)
#   after corrupt_step:  + corrupted (CorruptedSample | None)
#   after judge_step:    + judge_fields (dict | None)
#   output of build_step: final record dict | None
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 1 – corruption
# ---------------------------------------------------------------------------

def _corrupt(state: dict) -> dict:
    """Try each corruptor in order; attach the first successful result."""
    sample = state["sample"]
    for corruptor in state["corruptors"]:
        result = corruptor.corrupt(sample.question)
        if result is not None:
            log.info("[%s] corrupted via %s: %r", sample.sample_id, type(corruptor).__name__, result.corrupted_question)
            return {**state, "corrupted": result}
    log.info("[%s] no corruptor matched — dropped", sample.sample_id)
    return {**state, "corrupted": None}


# ---------------------------------------------------------------------------
# Step 2 – judge verification
# ---------------------------------------------------------------------------

def _judge(state: dict) -> dict:
    corrupted = state.get("corrupted")
    judge: Optional[LLMJudge] = state.get("judge")

    if corrupted is None:
        return {**state, "judge_fields": None}

    ctype = corrupted.corruption_type.value
    cdetail = corrupted.corruption_detail

    if judge is None:
        return {**state, "judge_fields": {
            "corrupted_question": corrupted.corrupted_question,
            "corruption_type": ctype,
            "corruption_detail": cdetail,
            "judge_verified": None,
        }}

    result: JudgeResult = judge.evaluate(corrupted.corrupted_question, state["sample"].document_path)
    log.info("[%s] judge → verdict=%s confidence=%.2f", state["sample"].sample_id, result.verdict, result.confidence)

    if result.verdict != "unanswerable":
        log.info("[%s] dropped (judge: answerable)", state["sample"].sample_id)
        return {**state, "judge_fields": None}

    return {**state, "judge_fields": {
        "corrupted_question": corrupted.corrupted_question,
        "corruption_type": ctype,
        "corruption_detail": cdetail,
        "judge_verified": True,
    }}


# ---------------------------------------------------------------------------
# Step 3 – assemble final record
# ---------------------------------------------------------------------------

def _build_record(state: dict) -> Optional[dict]:
    """Return the complete record dict, or None if the sample should be dropped."""
    if state.get("judge_fields") is None:
        return None
    sample = state["sample"]
    return {
        "sample_id": sample.sample_id,
        "document_path": sample.document_path,
        "original_question": sample.question,
        "original_answer": sample.answer,
        "page_index": sample.page_index,
        "metadata": sample.metadata,
        **state["judge_fields"],
    }


# ---------------------------------------------------------------------------
# Composed LCEL chain
# ---------------------------------------------------------------------------

_process_sample = (
    RunnableLambda(_corrupt)
    | RunnableLambda(_judge)
    | RunnableLambda(_build_record)
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset: str,
    data_dir: str,
    output_dir: str,
    config: dict,
    use_judge: bool = True,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    import inspect
    loader_cls = LOADERS[dataset]
    loader_kwargs = {
        k: v for k, v in config.get("loader", {}).items()
        if k in inspect.signature(loader_cls.__init__).parameters
    }
    loader = loader_cls(data_dir, **loader_kwargs)

    dist_cfg = config.get("corruption", {}).get("distribution", {})
    corruptor_map = {
        "nlp_entity": NLPEntityCorruptor,
        "element": ElementCorruptor,
        "layout": LayoutCorruptor,
    }
    dist_keys = [k for k in corruptor_map if k in dist_cfg]
    dist_weights = [dist_cfg[k] for k in dist_keys]
    corruptors = {k: corruptor_map[k](seed=seed) for k in dist_keys}
    qc = config.get("quality_check", {})
    judge = LLMJudge(
        model=qc.get("judge_model", "gemini-2.0-flash"),
        confidence_threshold=qc.get("confidence_threshold", 0.5),
        base_url=qc.get("judge_base_url") or None,
        max_retries=qc.get("max_retries", 3),
        max_tokens=qc.get("max_tokens", 2048),
    ) if use_judge else None

    log.info("Starting corruption pipeline: dataset=%s max_samples=%s judge=%s",
             dataset, config.get("corruption", {}).get("max_samples", "all"), "enabled" if judge else "disabled")

    all_samples = list(loader.load())
    max_samples = config.get("corruption", {}).get("max_samples", -1)
    if max_samples and max_samples > 0:
        all_samples = all_samples[:max_samples]
    log.info("Loaded %d samples", len(all_samples))

    output_path = Path(output_dir) / f"{dataset}_corrupted.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, sample in enumerate(all_samples):
        # Pick corruptor type by configured distribution; fall back to others on no match
        preferred = rng.choices(dist_keys, weights=dist_weights, k=1)[0]
        ordered = [preferred] + [k for k in dist_keys if k != preferred]
        shuffled = [corruptors[k] for k in ordered]
        state = {"sample": sample, "corruptors": shuffled, "judge": judge}
        record: Optional[dict] = _process_sample.invoke(state)
        if record is not None:
            results.append(record)
            log.info("[%s] ✓ kept [%s] %r → %r (total: %d)",
                     sample.sample_id,
                     record.get("corruption_type", "?"),
                     record.get("original_question"),
                     record.get("corrupted_question"),
                     len(results))
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        if (i + 1) % 50 == 0:
            log.info("Progress: %d/%d processed, %d kept", i + 1, len(all_samples), len(results))

    log.info("Done: %d/%d samples kept → %s", len(results), len(all_samples), output_path)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(LOADERS), required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="data/corrupted")
    parser.add_argument("--config", default="configs/dataset_config.yaml")
    parser.add_argument("--no_judge", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_pipeline(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
        use_judge=not args.no_judge,
        seed=args.seed,
    )


# ---------------------------------------------------------------------------
# Parallel runner – all datasets at once
# ---------------------------------------------------------------------------

def _run_dataset_worker(args: tuple) -> tuple[str, int]:
    """Top-level function so ProcessPoolExecutor can pickle it."""
    dataset, data_dir, output_dir, config_path, use_judge, seed = args
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [{dataset}] %(message)s",
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    records = run_pipeline(dataset, data_dir, output_dir, config, use_judge, seed)
    return dataset, len(records)


def main_all():
    """Run corruption pipeline for every dataset found in --base_dir in parallel."""
    parser = argparse.ArgumentParser(
        description="Corrupt all datasets under data/raw/ in parallel."
    )
    parser.add_argument("--base_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/corrupted")
    parser.add_argument("--config", default="configs/dataset_config.yaml")
    parser.add_argument("--no_judge", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base = Path(args.base_dir)
    jobs = [
        (ds, str(base / ds), args.output_dir, args.config, not args.no_judge, args.seed)
        for ds in LOADERS
        if (base / ds).exists()
    ]
    if not jobs:
        log.error("No dataset directories found under %s", args.base_dir)
        return

    log.info("Launching %d pipelines in parallel: %s", len(jobs), [j[0] for j in jobs])

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(_run_dataset_worker, job): job[0] for job in jobs}
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                _, count = future.result()
                log.info("✓ %s finished: %d records", dataset, count)
            except Exception as exc:
                log.error("✗ %s failed: %s", dataset, exc)


if __name__ == "__main__":
    main()
