"""Run mitigation experiments (few-shot, chain-of-thought, knowledge-injection, rag,
finetuning).

Usage:
    python -m src.mitigation.run_mitigation \
        --corrupted_dataset data/corrupted/docvqa_corrupted.json \
        --baseline_results results/benchmark/benchmark_results.json \
        --config configs/mitigation_config.yaml

To run only fine-tuning (requires GPU + unsloth):
    python -m src.mitigation.run_mitigation ... --strategies finetuning
"""

import argparse
import json
from pathlib import Path

import mlflow
import yaml

from .evaluation import evaluate_strategy
from .registry import STRATEGIES


def run_mitigation(
    corrupted_dataset_path: str,
    baseline_results_path: str,
    config: dict,
    output_dir: str,
    strategies: list[str] | None = None,
) -> None:
    with open(corrupted_dataset_path) as f:
        dataset: list[dict] = json.load(f)

    baseline_metrics: dict = {}
    if Path(baseline_results_path).exists():
        with open(baseline_results_path) as f:
            baseline_data = json.load(f)
        baseline_metrics = baseline_data.get("metrics", {})

    requested = strategies or config.get("strategies", list(STRATEGIES.keys()))
    model_cfg = config.get("model", {})
    model_id = model_cfg.get("model_id", "unknown")
    mlflow.set_experiment("mitigation")

    results: dict = {}

    prompt_strategy_names = [s for s in requested if s in STRATEGIES]
    if prompt_strategy_names:
        model = _load_model(model_cfg)
        for name in prompt_strategy_names:
            strategy_cls = STRATEGIES[name]
            strategy_cfg = config.get(name, {})
            strategy = strategy_cls(strategy_cfg)
            strategy.prepare(dataset, model)
            results[name] = evaluate_strategy(
                strategy=strategy,
                dataset=dataset,
                model=model,
                baseline_metrics=baseline_metrics,
                model_id=model_id,
                corrupted_dataset_path=corrupted_dataset_path,
            )

    if "finetuning" in requested:
        from .strategies.finetuning import FinetuningConfig, finetune
        ft_config = FinetuningConfig(**config.get("finetuning", {}))
        results["finetuning"] = {"metrics": finetune(dataset, ft_config)}

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted_dataset", required=True)
    parser.add_argument("--baseline_results", required=True)
    parser.add_argument("--config", default="configs/mitigation_config.yaml")
    parser.add_argument("--output_dir", default="results/mitigation")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Override strategies from config (e.g. --strategies rag few_shot)",
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
