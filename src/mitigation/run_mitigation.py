"""Part 3: run mitigation experiments (few-shot, chain-of-thought, knowledge-injection,
finetuning).

Usage:
    python -m src.mitigation.run_mitigation \
        --corrupted_dataset data/corrupted/docvqa_corrupted.json \
        --baseline_results results/benchmark/benchmark_results.json \
        --config configs/mitigation_config.yaml \
        --output_dir results/mitigation

To run only the fine-tuning strategy (requires GPU + unsloth):
    python -m src.mitigation.run_mitigation ... --strategies finetuning
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..benchmark.evaluation.metrics import compute_metrics, compute_per_type_metrics, plot_confusion_matrix
from .strategies.few_shot import build_few_shot_prompt
from .strategies.chain_of_thought import build_cot_prompt
from .strategies.knowledge_injection import DocumentMetadata, build_knowledge_injection_prompt
from .strategies.finetuning import FinetuningConfig, finetune


# ---------------------------------------------------------------------------
# Prompt-based strategies (question → prompt string)
# ---------------------------------------------------------------------------

_PROMPT_STRATEGIES: dict[str, object] = {
    "few_shot": lambda q, _: build_few_shot_prompt(q),
    "chain_of_thought": lambda q, _: build_cot_prompt(q),
    "knowledge_injection": lambda q, meta: build_knowledge_injection_prompt(
        q, DocumentMetadata()
    ),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_mitigation(
    corrupted_dataset_path: str,
    baseline_results_path: str,
    config: dict,
    output_dir: str,
    strategies: list[str] | None = None,
):
    with open(corrupted_dataset_path) as f:
        dataset: list[dict] = json.load(f)

    baseline_metrics: dict = {}
    if Path(baseline_results_path).exists():
        with open(baseline_results_path) as f:
            baseline_data = json.load(f)
        baseline_metrics = baseline_data.get("metrics", {})

    requested = strategies or config.get("strategies", list(_PROMPT_STRATEGIES.keys()))
    results = {}
    model_cfg = config.get("model", {})
    model_id = model_cfg.get("model_id", "unknown")
    mlflow.set_experiment("mitigation")

    # --- Prompt-based strategies ---
    prompt_strategies = [s for s in requested if s in _PROMPT_STRATEGIES]
    if prompt_strategies:
        model = _load_model(model_cfg)
        for strategy_name in prompt_strategies:
            prompt_fn = _PROMPT_STRATEGIES[strategy_name]
            preds, labels, records = [], [], []

            for item in dataset:
                prompt = prompt_fn(item["corrupted_question"], None)
                result = model.predict_unanswerable(
                    document_path=item["document_path"],
                    question=item["corrupted_question"],
                    prompt_template=prompt,
                )
                preds.append(result.predicted_unanswerable)
                labels.append(True)
                records.append({
                    "sample_id": item["sample_id"],
                    "strategy": strategy_name,
                    "predicted_unanswerable": result.predicted_unanswerable,
                    "label_unanswerable": True,
                    "raw_response": result.raw_response,
                    "corruption_type": item.get("corruption_type", "unknown"),
                })

            metrics = compute_metrics(labels, preds)
            print(f"\n[{strategy_name}] {metrics}")
            results[strategy_name] = {"records": records, "metrics": metrics.__dict__}

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model = model_id.replace("/", "_")
            with mlflow.start_run(run_name=f"{strategy_name}_{safe_model}_{timestamp}"):
                mlflow.log_params({
                    "strategy": strategy_name,
                    "model_id": model_id,
                    "dataset_path": corrupted_dataset_path,
                    "num_samples": len(dataset),
                })
                delta_f1 = metrics.f1 - baseline_metrics.get("f1", 0.0)
                delta_mcc = metrics.mcc - baseline_metrics.get("mcc", 0.0)
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
                })
                per_type = compute_per_type_metrics(records)
                for ctype, tm in per_type.items():
                    mlflow.log_metric(f"f1_{ctype}", tm.f1)
                fig = plot_confusion_matrix(metrics, title=f"{strategy_name} — {model_id}")
                mlflow.log_figure(fig, f"confusion_matrix_{strategy_name}_{safe_model}.png")
                plt.close(fig)

    # --- Fine-tuning strategy ---
    if "finetuning" in requested:
        ft_cfg_dict = config.get("finetuning", {})
        ft_config = FinetuningConfig(**ft_cfg_dict)
        ft_metrics = finetune(dataset, ft_config)
        results["finetuning"] = {"metrics": ft_metrics}

    # --- Persist ---
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "mitigation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out / 'mitigation_results.json'}")


def _load_model(model_cfg: dict):
    """Instantiate the benchmark vision model from config."""
    backend = model_cfg.get("backend", "")
    model_id = model_cfg.get("model_id", "")
    if backend == "mistral":
        from ..benchmark.models.mistral_model import MistralModel
        return MistralModel(model_id=model_id)
    if backend == "google":
        from ..benchmark.models.google_model import GoogleModel
        return GoogleModel(model_id=model_id)
    if backend == "openrouter":
        from ..benchmark.models.openrouter_model import OpenRouterModel
        return OpenRouterModel(model_id=model_id)
    if backend == "llama_cpp":
        from ..benchmark.models.llama_cpp_model import LlamaCppModel
        return LlamaCppModel(model_id=model_id)
    if backend == "vllm":
        from ..benchmark.models.vllm_model import VllmModel
        return VllmModel(
            base_url=model_cfg.get("base_url", "http://localhost:8083/v1"),
            model_id=model_id,
            api_key=model_cfg.get("api_key", "local"),
            max_tokens=model_cfg.get("max_tokens", 256),
        )
    raise ValueError(f"Unknown benchmark model backend: '{backend}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted_dataset", required=True)
    parser.add_argument("--baseline_results", required=True)
    parser.add_argument("--config", default="configs/mitigation_config.yaml")
    parser.add_argument("--output_dir", default="results/mitigation")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Override strategies from config (e.g. --strategies finetuning few_shot)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_mitigation(
        corrupted_dataset_path=args.corrupted_dataset,
        baseline_results_path=args.baseline_results,
        config=config,
        output_dir=args.output_dir,
        strategies=args.strategies,
    )


if __name__ == "__main__":
    main()
