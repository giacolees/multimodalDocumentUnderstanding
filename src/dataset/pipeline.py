"""Main corruption pipeline for Part 1.

Orchestrated with LangChain LCEL: each processing phase is a RunnableLambda
and samples flow through the composed chain via .invoke().

Usage:
    python -m src.part1_dataset.pipeline \
        --dataset docvqa \
        --data_dir data/raw/docvqa \
        --output_dir data/corrupted \
        --config configs/dataset_config.yaml
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import yaml
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
            return {**state, "corrupted": result}
    return {**state, "corrupted": None}


# ---------------------------------------------------------------------------
# Step 2 – judge verification (with one suggestion retry)
# ---------------------------------------------------------------------------

def _judge(state: dict) -> dict:
    """Verify the corrupted question; try the judge's suggestion on failure."""
    corrupted = state.get("corrupted")
    judge: Optional[LLMJudge] = state.get("judge")

    if corrupted is None:
        return {**state, "judge_fields": None}

    ctype = corrupted.corruption_type.value
    cdetail = corrupted.corruption_detail

    if judge is None:
        # Judge disabled — pass through without verification
        return {**state, "judge_fields": {
            "corrupted_question": corrupted.corrupted_question,
            "corruption_type": ctype,
            "corruption_detail": cdetail,
            "judge_verified": None,
            "judge_reason": None,
            "judge_suggested": None,
        }}

    result: JudgeResult = judge.evaluate(corrupted.corrupted_question, state["sample"].document_path)

    if result.verdict == "unanswerable":
        return {**state, "judge_fields": {
            "corrupted_question": corrupted.corrupted_question,
            "corruption_type": ctype,
            "corruption_detail": cdetail,
            "judge_verified": True,
            "judge_reason": result.reason,
            "judge_suggested": False,
        }}

    # Judge rejected — try its suggestion once
    if result.suggested_question:
        retry: JudgeResult = judge.evaluate(result.suggested_question, state["sample"].document_path)
        if retry.verdict == "unanswerable":
            return {**state, "judge_fields": {
                "corrupted_question": result.suggested_question,
                "corruption_type": ctype,
                "corruption_detail": f"{cdetail} [judge-revised]",
                "judge_verified": True,
                "judge_reason": retry.reason,
                "judge_suggested": True,
            }}

    return {**state, "judge_fields": None}


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
    loader_cls = LOADERS[dataset]
    loader = loader_cls(data_dir, **config.get("loader", {}))

    corruptors = [C(seed=seed) for C in CORRUPTORS]
    judge = LLMJudge() if use_judge else None

    results = []
    for sample in loader.load():
        shuffled = rng.sample(corruptors, len(corruptors))
        state = {"sample": sample, "corruptors": shuffled, "judge": judge}
        record: Optional[dict] = _process_sample.invoke(state)
        if record is not None:
            results.append(record)

    output_path = Path(output_dir) / f"{dataset}_corrupted.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} corrupted samples → {output_path}")
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


if __name__ == "__main__":
    main()
