"""Shared evaluation loop for all prompt-based mitigation strategies."""

from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from ..benchmark.evaluation.metrics import (
    compute_metrics,
    compute_per_type_metrics,
    plot_confusion_matrix,
)

if TYPE_CHECKING:
    from .strategies.base import MitigationStrategy
    from ..benchmark.models.base_model import BaseVisionModel


def evaluate_strategy(
    strategy: "MitigationStrategy",
    dataset: list[dict],
    model: "BaseVisionModel",
    baseline_metrics: dict,
    model_id: str,
    corrupted_dataset_path: str,
) -> dict:
    """Run the per-item inference loop, compute metrics, log to MLflow, return results dict."""
    preds: list[bool] = []
    labels: list[bool] = []
    records: list[dict] = []
    inference_times: list[float] = []
    _sample_prompt: str = ""

    for item in dataset:
        prompt = strategy.build_prompt(item, model)
        if not _sample_prompt:
            _sample_prompt = prompt
        t0 = time.perf_counter()
        result = model.predict_unanswerable(
            document_path=item["document_path"],
            question=item["corrupted_question"],
            prompt_template=prompt,
        )
        elapsed = time.perf_counter() - t0
        inference_times.append(elapsed)
        preds.append(result.predicted_unanswerable)
        labels.append(True)
        records.append({
            "sample_id": item["sample_id"],
            "strategy": strategy.name,
            "predicted_unanswerable": result.predicted_unanswerable,
            "label_unanswerable": True,
            "raw_response": result.raw_response,
            "corruption_type": item.get("corruption_type", "unknown"),
            "inference_time_s": elapsed,
        })

    metrics = compute_metrics(labels, preds)
    print(f"\n[{strategy.name}] {metrics}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_id.replace("/", "_")

    with mlflow.start_run(run_name=f"{strategy.name}_{safe_model}_{timestamp}"):
        mlflow.set_tags({
            "strategy": strategy.name,
            "model": model_id,
            "dataset": Path(corrupted_dataset_path).stem,
        })
        mlflow.log_params({
            "strategy": strategy.name,
            "model_id": model_id,
            "dataset_path": corrupted_dataset_path,
            "num_samples": len(dataset),
        })
        _df = pd.read_json(corrupted_dataset_path)
        _targets = "is_unanswerable" if "is_unanswerable" in _df.columns else None
        ds = mlflow.data.from_pandas(
            _df,
            name=Path(corrupted_dataset_path).stem,
            targets=_targets,
        )
        mlflow.log_input(ds, context="evaluation")

        if _sample_prompt:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False,
                prefix=f"prompt_{strategy.name}_",
            ) as tmp:
                tmp.write(_sample_prompt)
            try:
                mlflow.log_artifact(tmp.name, artifact_path="prompts")
            finally:
                Path(tmp.name).unlink(missing_ok=True)

        delta_f1 = metrics.f1 - baseline_metrics.get("f1", 0.0)
        delta_mcc = metrics.mcc - baseline_metrics.get("mcc", 0.0)
        delta_precision = metrics.precision - baseline_metrics.get("precision", 0.0)
        delta_recall = metrics.recall - baseline_metrics.get("recall", 0.0)
        delta_specificity = metrics.specificity - baseline_metrics.get("specificity", 0.0)
        delta_balanced_accuracy = (
            metrics.balanced_accuracy - baseline_metrics.get("balanced_accuracy", 0.0)
        )
        mlflow.log_metrics({
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "tp": float(metrics.tp),
            "fp": float(metrics.fp),
            "tn": float(metrics.tn),
            "fn": float(metrics.fn),
            "specificity": metrics.specificity,
            "balanced_accuracy": metrics.balanced_accuracy,
            "mcc": metrics.mcc,
            "delta_f1": delta_f1,
            "delta_mcc": delta_mcc,
            "delta_precision": delta_precision,
            "delta_recall": delta_recall,
            "delta_specificity": delta_specificity,
            "delta_balanced_accuracy": delta_balanced_accuracy,
            **(
                {
                    "inference_time_mean_s": float(np.mean(inference_times)),
                    "inference_time_median_s": float(np.median(inference_times)),
                    "inference_time_p95_s": float(np.percentile(inference_times, 95)),
                    "inference_time_total_s": sum(inference_times),
                    "throughput_samples_per_s": len(inference_times) / sum(inference_times),
                }
                if inference_times
                else {}
            ),
        })

        per_type = compute_per_type_metrics(records)
        baseline_per_type = baseline_metrics.get("per_type", {})
        for ctype, tm in per_type.items():
            base_f1 = (
                baseline_per_type.get(ctype, {}).get("f1", 0.0)
                if isinstance(baseline_per_type, dict)
                else 0.0
            )
            mlflow.log_metrics({
                f"f1_{ctype}": tm.f1,
                f"precision_{ctype}": tm.precision,
                f"recall_{ctype}": tm.recall,
                f"specificity_{ctype}": tm.specificity,
                f"mcc_{ctype}": tm.mcc,
                f"delta_f1_{ctype}": tm.f1 - base_f1,
            })

        fig = plot_confusion_matrix(metrics, title=f"{strategy.name} — {model_id}")
        mlflow.log_figure(fig, f"confusion_matrix_{strategy.name}_{safe_model}.png")
        plt.close(fig)

        if per_type:
            ctypes = list(per_type.keys())
            strategy_f1s = [per_type[ct].f1 for ct in ctypes]
            base_f1s = [
                baseline_per_type.get(ct, {}).get("f1", 0.0)
                if isinstance(baseline_per_type, dict)
                else 0.0
                for ct in ctypes
            ]
            x = range(len(ctypes))
            fig3, ax3 = plt.subplots(figsize=(max(5, len(ctypes) * 2), 4))
            ax3.bar([i - 0.2 for i in x], base_f1s, width=0.4, label="baseline", color="lightgray")
            ax3.bar(
                [i + 0.2 for i in x], strategy_f1s, width=0.4,
                label=strategy.name, color="steelblue",
            )
            ax3.set_xticks(list(x))
            ax3.set_xticklabels(ctypes)
            ax3.set_ylim(0, 1.1)
            ax3.set_ylabel("F1")
            ax3.set_title(f"F1 by type: {strategy.name} vs baseline")
            ax3.legend()
            plt.tight_layout()
            mlflow.log_figure(fig3, f"f1_by_type_{strategy.name}_{safe_model}.png")
            plt.close(fig3)

    return {"records": records, "metrics": metrics.__dict__}
