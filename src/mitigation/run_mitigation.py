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
from pathlib import Path

import yaml

from ..benchmark.evaluation.metrics import compute_metrics
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

    requested = strategies or config.get("strategies", list(_PROMPT_STRATEGIES.keys()))
    results = {}

    # --- Prompt-based strategies ---
    prompt_strategies = [s for s in requested if s in _PROMPT_STRATEGIES]
    if prompt_strategies:
        model = _load_model(config["model"])
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
                    "raw_response": result.raw_response,
                })

            metrics = compute_metrics(labels, preds)
            print(f"\n[{strategy_name}] {metrics}")
            results[strategy_name] = {"records": records, "metrics": metrics.__dict__}

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
