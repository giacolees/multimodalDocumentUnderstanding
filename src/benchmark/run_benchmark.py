"""Run benchmarking experiments.

Usage:
    python -m src.benchmark.run_benchmark --config configs/benchmark_config.yaml

corrupted_dataset and output_dir are set in the config file.
"""

import argparse
import csv
import io
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models.mistral_model import MistralModel
from .models.openrouter_model import OpenRouterModel
from .models.google_model import GoogleModel
from .models.llama_cpp_model import LlamaCppModel
from .models.vllm_model import VllmModel
from .evaluation.metrics import compute_metrics, BenchmarkMetrics, compute_per_type_metrics, plot_confusion_matrix


_BASELINE_PROMPT = (
    "Look at the document image and answer the following question.\n"
    "If the question cannot be answered from the document, respond with exactly: UNANSWERABLE\n"
    "Otherwise provide the answer.\n\n"
    "Question: {question}"
)


def load_model(model_cfg: dict):
    backend = model_cfg["backend"]
    if backend == "mistral":
        return MistralModel(model_id=model_cfg.get("model_id", "pixtral-12b-2409"))
    if backend == "openrouter":
        return OpenRouterModel(
            model_id=model_cfg.get("model_id", "google/gemini-2.0-flash-exp"),
            site_url=model_cfg.get("site_url", ""),
            site_name=model_cfg.get("site_name", ""),
        )
    if backend == "google":
        return GoogleModel(model_id=model_cfg.get("model_id", "gemini-2.0-flash"))
    if backend == "llama_cpp":
        return LlamaCppModel(
            mode=model_cfg.get("mode", "server"),
            model_path=model_cfg.get("model_path"),
            clip_model_path=model_cfg.get("clip_model_path"),
            n_ctx=model_cfg.get("n_ctx", 4096),
            n_gpu_layers=model_cfg.get("n_gpu_layers", -1),
            base_url=model_cfg.get("base_url", "http://localhost:8080/v1"),
            model_id=model_cfg.get("model_id", "local"),
        )
    if backend == "vllm":
        return VllmModel(
            base_url=model_cfg.get("base_url", "http://localhost:8083/v1"),
            model_id=model_cfg.get("model_id", "google/gemma-4-12b-it"),
            api_key=model_cfg.get("api_key", "local"),
            max_tokens=model_cfg.get("max_tokens", 256),
            image_placeholder=model_cfg.get("image_placeholder", ""),
            max_image_pixels=model_cfg.get("max_image_pixels", 0),
            stop_sequences=model_cfg.get("stop_sequences"),
        )
    raise ValueError(f"Unknown backend: {backend}")


def run_benchmark(
    corrupted_dataset_path: str,
    config: dict,
    output_dir: str,
    prompt_template: str = _BASELINE_PROMPT,
):
    with open(corrupted_dataset_path) as f:
        dataset = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset_name = Path(corrupted_dataset_path).stem
    mlflow.set_experiment(config.get("mlflow_experiment", "benchmark"))

    for model_cfg in config["models"]:
        model = load_model(model_cfg)
        model_name = model.name()
        safe_name = model_name.replace("/", "_")
        results_path = out / f"{safe_name}_benchmark_result.json"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        existing: dict = {}
        if results_path.exists():
            with open(results_path) as f:
                existing = json.load(f)

        existing_records: list[dict] = existing.get("records", [])
        done_ids = {r["sample_id"] for r in existing_records}
        remaining = [item for item in dataset if item["sample_id"] not in done_ids]

        if not remaining:
            print(f"\n[{model_name}] all {len(dataset)} samples already done, skipping")
            continue

        with mlflow.start_run(run_name=f"{safe_name}_{dataset_name}_{timestamp}"):
            mlflow.set_tags({
                "model": model_name,
                "backend": model_cfg["backend"],
                "dataset": dataset_name,
            })
            mlflow.log_params({
                "model_id": model_name,
                "backend": model_cfg["backend"],
                "dataset_path": corrupted_dataset_path,
                "num_samples": len(dataset),
                "label_unanswerable_rate": round(
                    sum(1 for it in dataset if it.get("is_unanswerable", True)) / len(dataset), 4
                ) if dataset else 0.0,
            })
            import pandas as pd
            ds = mlflow.data.from_pandas(pd.read_json(corrupted_dataset_path), name=dataset_name, targets="is_unanswerable")
            mlflow.log_input(ds, context="benchmark")

            if done_ids:
                print(f"\n[{model_name}] resuming: {len(done_ids)} done, {len(remaining)} remaining")

            records = list(existing_records)
            _log_every = max(1, len(remaining) // 10)  # ~10 checkpoints during inference
            for _step_i, item in enumerate(remaining):
                # mixed benchmark datasets use "question" + "is_unanswerable";
                # legacy corrupted-only datasets use "corrupted_question" with implied True
                question = item["question"] if "question" in item else item["corrupted_question"]
                label: bool = item.get("is_unanswerable", True)
                t0 = time.perf_counter()
                result = model.predict_unanswerable(
                    document_path=item["document_path"],
                    question=question,
                    prompt_template=prompt_template,
                )
                result.inference_time_s = time.perf_counter() - t0
                result.sample_id = item["sample_id"]
                records.append({
                    "sample_id": item["sample_id"],
                    "predicted_unanswerable": result.predicted_unanswerable,
                    "label_unanswerable": label,
                    "raw_response": result.raw_response,
                    "corruption_type": item["corruption_type"],
                    "inference_time_s": result.inference_time_s,
                    "response_length": len(result.raw_response or ""),
                    "skipped": result.skipped,
                })
                # save after every sample so a crash loses at most one result
                _scored = [r for r in records if not r.get("skipped")]
                _preds = [r["predicted_unanswerable"] for r in _scored]
                _labels = [r["label_unanswerable"] for r in _scored]
                _m = compute_metrics(_labels, _preds)
                with open(results_path, "w") as f:
                    json.dump({"records": records, "metrics": _m.__dict__}, f, indent=2)
                # log rolling metrics as steps so MLflow renders trend charts
                if (_step_i + 1) % _log_every == 0 or _step_i + 1 == len(remaining):
                    _global_step = len(done_ids) + _step_i + 1
                    mlflow.log_metrics({
                        "rolling_accuracy": _m.accuracy,
                        "rolling_f1": _m.f1,
                        "rolling_mcc": _m.mcc,
                        "rolling_precision": _m.precision,
                        "rolling_recall": _m.recall,
                    }, step=_global_step)

            scored = [r for r in records if not r.get("skipped")]
            n_skipped = len(records) - len(scored)
            preds = [r["predicted_unanswerable"] for r in scored]
            labels = [r["label_unanswerable"] for r in scored]
            metrics = compute_metrics(labels, preds)
            print(f"\n[{model_name}] {metrics}" + (f"  ({n_skipped} skipped)" if n_skipped else ""))
            with open(results_path, "w") as f:
                json.dump({"records": records, "metrics": metrics.__dict__}, f, indent=2)
            print(f"Results saved → {results_path}")

            inference_times = [r["inference_time_s"] for r in scored if "inference_time_s" in r]
            response_lengths = [r["response_length"] for r in scored if "response_length" in r]
            pred_unanswerable_rate = sum(preds) / len(preds) if preds else 0.0
            mlflow.log_metrics({
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "tp": float(metrics.tp),
                "fp": float(metrics.fp),
                "tn": float(metrics.tn),
                "fn": float(metrics.fn),
                "n_skipped": float(n_skipped),
                "specificity": metrics.specificity,
                "balanced_accuracy": metrics.balanced_accuracy,
                "mcc": metrics.mcc,
                "pred_unanswerable_rate": pred_unanswerable_rate,
                **({"inference_time_mean_s": float(np.mean(inference_times)),
                    "inference_time_median_s": float(np.median(inference_times)),
                    "inference_time_p95_s": float(np.percentile(inference_times, 95)),
                    "inference_time_total_s": sum(inference_times),
                    "inference_time_max_s": max(inference_times),
                    "throughput_samples_per_s": len(inference_times) / sum(inference_times),
                } if inference_times else {}),
                **({"response_length_mean": float(np.mean(response_lengths)),
                    "response_length_median": float(np.median(response_lengths)),
                } if response_lengths else {}),
            })

            per_type = compute_per_type_metrics(scored)
            for ctype, tm in per_type.items():
                mlflow.log_metrics({
                    f"f1_{ctype}": tm.f1,
                    f"precision_{ctype}": tm.precision,
                    f"recall_{ctype}": tm.recall,
                    f"specificity_{ctype}": tm.specificity,
                    f"mcc_{ctype}": tm.mcc,
                    f"balanced_accuracy_{ctype}": tm.balanced_accuracy,
                })

            fig = plot_confusion_matrix(metrics, title=model_name)
            mlflow.log_figure(fig, f"confusion_matrix_{safe_name}.png")
            plt.close(fig)

            # per-type F1 bar chart
            if per_type:
                fig2, ax2 = plt.subplots(figsize=(max(4, len(per_type) * 1.5), 4))
                ctypes = list(per_type.keys())
                vals = [per_type[ct].f1 for ct in ctypes]
                bars = ax2.bar(ctypes, vals, color="steelblue")
                ax2.bar_label(bars, fmt="%.3f", padding=3)
                ax2.set_ylim(0, 1.1)
                ax2.set_ylabel("F1")
                ax2.set_title(f"F1 by corruption type — {model_name}")
                plt.tight_layout()
                mlflow.log_figure(fig2, f"f1_by_type_{safe_name}.png")
                plt.close(fig2)

            # FP / FN error-analysis CSV artifacts
            for error_type, condition in [("false_positives", lambda r: not r["label_unanswerable"] and r["predicted_unanswerable"]),
                                          ("false_negatives", lambda r: r["label_unanswerable"] and not r["predicted_unanswerable"])]:
                error_records = [r for r in records if condition(r)]
                if error_records:
                    buf = io.StringIO()
                    writer = csv.DictWriter(buf, fieldnames=["sample_id", "corruption_type", "raw_response", "inference_time_s"])
                    writer.writeheader()
                    writer.writerows([{k: r.get(k, "") for k in writer.fieldnames} for r in error_records])
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, prefix=f"{error_type}_") as tmp:
                        tmp.write(buf.getvalue())
                        mlflow.log_artifact(tmp.name, artifact_path="error_analysis")

            mlflow.log_artifact(str(results_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run_benchmark(
        config["corrupted_dataset"],
        config,
        config["output_dir"],
    )


if __name__ == "__main__":
    main()
