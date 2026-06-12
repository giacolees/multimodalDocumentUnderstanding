"""Main corruption pipeline.

Usage:
    python -m src.dataset.pipeline --config configs/dataset_config.yaml

dataset, data_dir, and output_dir are set in the config file.
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import yaml

log = logging.getLogger(__name__)

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
        return {**state, "judge_fields": None, "judge_verdict": result.verdict, "judge_confidence": result.confidence}

    return {**state, "judge_fields": {
        "corrupted_question": corrupted.corrupted_question,
        "corruption_type": ctype,
        "corruption_detail": cdetail,
        "judge_verified": True,
        "judge_confidence": result.confidence,
    }, "judge_verdict": result.verdict, "judge_confidence": result.confidence}


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
    judge_verdicts: list[dict] = []
    for i, sample in enumerate(all_samples):
        # Pick corruptor type by configured distribution; fall back to others on no match
        preferred = rng.choices(dist_keys, weights=dist_weights, k=1)[0]
        ordered = [preferred] + [k for k in dist_keys if k != preferred]
        shuffled = [corruptors[k] for k in ordered]
        state = {"sample": sample, "corruptors": shuffled, "judge": judge}
        # run steps individually to capture judge state for MLflow verdict table
        s = _corrupt(state)
        s = _judge(s)
        record = _build_record(s)
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
        if use_judge and s.get("judge_verdict") is not None:
            judge_verdicts.append({
                "sample_id": sample.sample_id,
                "corruption_type": s["corrupted"].corruption_type.value if s.get("corrupted") else "",
                "verdict": s["judge_verdict"],
                "confidence": round(s["judge_confidence"], 4),
                "kept": record is not None,
            })
        if (i + 1) % 50 == 0:
            log.info("Progress: %d/%d processed, %d kept", i + 1, len(all_samples), len(results))

    log.info("Done: %d/%d samples kept → %s", len(results), len(all_samples), output_path)

    # --- MLflow tracking ---
    import csv, io, tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow.set_experiment("dataset-corruption")
    with mlflow.start_run(run_name=f"{dataset}_{timestamp}"):
        qc = config.get("quality_check", {})
        mlflow.set_tags({
            "dataset": dataset,
            "use_judge": str(use_judge),
        })
        mlflow.log_params({
            "dataset": dataset,
            "max_samples": config.get("corruption", {}).get("max_samples", -1),
            "window_size": config.get("loader", {}).get("window_size", 1),
            "corruption_types": ",".join(dist_keys),
            "use_judge": use_judge,
            "judge_model": qc.get("judge_model", "gemini-2.0-flash") if use_judge else None,
            "judge_confidence_threshold": qc.get("confidence_threshold", 0.5) if use_judge else None,
            "judge_base_url": qc.get("judge_base_url") if use_judge else None,
        })
        type_counts: dict[str, int] = {}
        for r in results:
            ct = r.get("corruption_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1
        judge_dropped = sum(1 for v in judge_verdicts if not v["kept"])
        judge_kept = sum(1 for v in judge_verdicts if v["kept"])
        dropped_no_match = len(all_samples) - len(results) - judge_dropped
        yield_rate = len(results) / len(all_samples) if all_samples else 0.0

        orig_lengths = [len(r.get("original_question", "")) for r in results]
        corr_lengths = [len(r.get("corrupted_question", "")) for r in results]

        mlflow.log_metrics({
            "total_samples": len(all_samples),
            "total_kept": len(results),
            "dropped_no_corruptor_match": dropped_no_match,
            "yield_rate": yield_rate,
            **{f"{ct}_count": cnt for ct, cnt in type_counts.items()},
            **({"judge_kept": judge_kept, "judge_dropped": judge_dropped,
                "judge_acceptance_rate": judge_kept / len(judge_verdicts) if judge_verdicts else 0.0,
                "judge_avg_confidence": sum(v["confidence"] for v in judge_verdicts) / len(judge_verdicts) if judge_verdicts else 0.0,
                "judge_min_confidence": min(v["confidence"] for v in judge_verdicts) if judge_verdicts else 0.0,
                "judge_max_confidence": max(v["confidence"] for v in judge_verdicts) if judge_verdicts else 0.0,
               } if use_judge else {}),
            **({"question_length_orig_mean": float(np.mean(orig_lengths)),
                "question_length_corr_mean": float(np.mean(corr_lengths)),
                "question_length_delta_mean": float(np.mean(corr_lengths)) - float(np.mean(orig_lengths)),
               } if orig_lengths else {}),
        })

        # corruption type distribution bar chart
        if type_counts:
            fig_bar, ax_bar = plt.subplots(figsize=(max(4, len(type_counts) * 1.5), 4))
            bars = ax_bar.bar(list(type_counts.keys()), list(type_counts.values()), color="steelblue")
            ax_bar.bar_label(bars, padding=3)
            ax_bar.set_ylabel("Count")
            ax_bar.set_title(f"Corruption type distribution — {dataset}")
            plt.tight_layout()
            mlflow.log_figure(fig_bar, "corruption_type_distribution.png")
            plt.close(fig_bar)

        # judge confidence histogram
        if use_judge and judge_verdicts:
            confidences = [v["confidence"] for v in judge_verdicts]
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
            ax_hist.hist(confidences, bins=20, color="steelblue", edgecolor="white")
            ax_hist.set_xlabel("Judge confidence")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title(f"Judge confidence distribution — {dataset}")
            plt.tight_layout()
            mlflow.log_figure(fig_hist, "judge_confidence_histogram.png")
            plt.close(fig_hist)

            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=["sample_id", "corruption_type", "verdict", "confidence", "kept"])
            writer.writeheader()
            writer.writerows(judge_verdicts)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, prefix="judge_verdicts_") as tmp:
                tmp.write(buf.getvalue())
                tmp_path = tmp.name
            mlflow.log_artifact(tmp_path, artifact_path="judge")

        if output_path.exists():
            mlflow.log_artifact(str(output_path))
            ds = mlflow.data.from_json(str(output_path), name=dataset, targets="corrupted_question")
            mlflow.log_input(ds, context="output")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_config.yaml")
    parser.add_argument("--no_judge", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_pipeline(
        dataset=config["dataset"],
        data_dir=config["data_dir"],
        output_dir=config["output_dir"],
        config=config,
        use_judge=not args.no_judge,
        seed=config["corruption"]["seed"],
    )


def main_all():
    """Run corruption pipeline for every dataset found under data_dir's parent."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_config.yaml")
    parser.add_argument("--no_judge", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    base = Path(config["data_dir"]).parent
    datasets = [ds for ds in LOADERS if (base / ds).exists()]
    if not datasets:
        log.error("No dataset directories found under %s", base)
        return

    for ds in datasets:
        records = run_pipeline(ds, str(base / ds), config["output_dir"], config, not args.no_judge, config["corruption"]["seed"])
        log.info("✓ %s finished: %d records", ds, len(records))


if __name__ == "__main__":
    main()
