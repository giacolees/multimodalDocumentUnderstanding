"""Part 2: run benchmarking experiments.

Usage:
    python -m src.benchmark.run_benchmark \
        --corrupted_dataset data/corrupted/docvqa_corrupted.json \
        --config configs/benchmark_config.yaml \
        --output_dir results/benchmark
"""

import argparse
import json
from pathlib import Path

import yaml

from .models.mistral_model import MistralModel
from .models.openrouter_model import OpenRouterModel
from .models.google_model import GoogleModel
from .models.llama_cpp_model import LlamaCppModel
from .evaluation.metrics import compute_metrics, BenchmarkMetrics


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
    raise ValueError(f"Unknown backend: {backend}")


def run_benchmark(
    corrupted_dataset_path: str,
    config: dict,
    output_dir: str,
    prompt_template: str = _BASELINE_PROMPT,
):
    with open(corrupted_dataset_path) as f:
        dataset = json.load(f)

    results_by_model: dict[str, list[dict]] = {}

    for model_cfg in config["models"]:
        model = load_model(model_cfg)
        preds, labels = [], []
        records = []

        for item in dataset:
            result = model.predict_unanswerable(
                document_path=item["document_path"],
                question=item["corrupted_question"],
                prompt_template=prompt_template,
            )
            result.sample_id = item["sample_id"]
            preds.append(result.predicted_unanswerable)
            labels.append(True)  # all corrupted samples are unanswerable
            records.append({
                "sample_id": item["sample_id"],
                "predicted_unanswerable": result.predicted_unanswerable,
                "raw_response": result.raw_response,
                "corruption_type": item["corruption_type"],
            })

        metrics = compute_metrics(labels, preds)
        print(f"\n[{model.name()}] {metrics}")
        results_by_model[model.name()] = {"records": records, "metrics": metrics.__dict__}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "benchmark_results.json", "w") as f:
        json.dump(results_by_model, f, indent=2)
    print(f"\nResults saved → {out / 'benchmark_results.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted_dataset", required=True)
    parser.add_argument("--config", default="configs/benchmark_config.yaml")
    parser.add_argument("--output_dir", default="results/benchmark")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run_benchmark(args.corrupted_dataset, config, args.output_dir)


if __name__ == "__main__":
    main()
