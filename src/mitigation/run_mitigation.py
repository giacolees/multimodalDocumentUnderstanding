"""Run mitigation experiments (few-shot, chain-of-thought, knowledge-injection, rag,
finetuning).

Usage:
    python -m src.mitigation.run_mitigation \
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


def _infer_benchmark_dir(corrupted_dataset_path: str) -> Path | None:
    """Derive the benchmark results directory from the corrupted dataset path.

    e.g. data/corrupted/docvqa_corrupted.json → results/benchmark_docvqa/
         data/corrupted/mp_docvqa_corrupted.json → results/benchmark_mp_docvqa/
    """
    stem = Path(corrupted_dataset_path).stem  # e.g. "docvqa_corrupted"
    dataset_name = stem.replace("_corrupted", "")  # e.g. "docvqa"
    candidate = Path("results") / f"benchmark_{dataset_name}"
    return candidate if candidate.is_dir() else None


def load_subset_baselines(
    corrupted_dataset_path: str,
    sample_ids: set[str],
) -> dict[str, dict]:
    """For each model benchmark result file, filter records to sample_ids and recompute metrics.

    Returns {model_name: {"metrics": dict, "n_matched": int, "n_total": int}}.
    """
    from ..benchmark.evaluation.metrics import compute_metrics

    bench_dir = _infer_benchmark_dir(corrupted_dataset_path)
    if bench_dir is None:
        return {}

    baselines: dict[str, dict] = {}
    for result_file in sorted(bench_dir.glob("*_benchmark_result.json")):
        model_name = result_file.stem.replace("_benchmark_result", "")
        with open(result_file) as f:
            data = json.load(f)

        records = data.get("records", [])
        subset = [r for r in records if r["sample_id"] in sample_ids]
        if not subset:
            continue

        y_true = [r["label_unanswerable"] for r in subset]
        y_pred = [r["predicted_unanswerable"] for r in subset]
        metrics = compute_metrics(y_true, y_pred)
        baselines[model_name] = {
            "metrics": metrics.__dict__,
            "n_matched": len(subset),
            "n_total": len(records),
        }

    return baselines


def _print_subset_baselines(baselines: dict[str, dict], n_subset: int) -> None:
    if not baselines:
        return
    print(f"\n{'─'*72}")
    print(f"Baseline (non-enhanced) performance on the {n_subset}-sample subset:")
    print(f"{'Model':<45} {'F1':>6} {'MCC':>6} {'Prec':>6} {'Rec':>6}  matched")
    print(f"{'─'*72}")
    for model, info in baselines.items():
        m = info["metrics"]
        print(
            f"{model:<45} {m['f1']:>6.3f} {m['mcc']:>6.3f}"
            f" {m['precision']:>6.3f} {m['recall']:>6.3f}"
            f"  {info['n_matched']}/{info['n_total']}"
        )
    print(f"{'─'*72}\n")


def save_baseline_subset(
    model_cfg: dict,
    sample_ids: set[str],
    corrupted_dataset_path: str,
    output_dir: str | Path,
) -> Path | None:
    """Find the benchmark result file for the current model and save the subset of records
    matching sample_ids to output_dir/baseline_subset_results.json."""
    bench_dir = _infer_benchmark_dir(corrupted_dataset_path)
    if bench_dir is None:
        return None

    backend = model_cfg.get("backend", "")
    model_id = model_cfg.get("model_id", "")
    expected_stem = f"{backend}:{model_id.replace('/', '_')}"
    result_file = bench_dir / f"{expected_stem}_benchmark_result.json"

    if not result_file.exists():
        # Fall back to searching for any file whose stem contains the model_id fragment.
        fragment = model_id.split("/")[-1]
        matches = list(bench_dir.glob(f"*{fragment}*_benchmark_result.json"))
        if not matches:
            return None
        result_file = matches[0]

    with open(result_file) as f:
        data = json.load(f)

    subset_records = [r for r in data.get("records", []) if r["sample_id"] in sample_ids]

    from ..benchmark.evaluation.metrics import compute_metrics
    y_true = [r["label_unanswerable"] for r in subset_records]
    y_pred = [r["predicted_unanswerable"] for r in subset_records]
    metrics = compute_metrics(y_true, y_pred).__dict__ if subset_records else {}

    out_path = Path(output_dir) / "baseline_subset_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {"source_file": str(result_file), "records": subset_records, "metrics": metrics},
            f, indent=2,
        )
    print(f"Baseline subset saved → {out_path}  ({len(subset_records)} records)")
    return out_path


def _expand_strategies(requested: list[str], config: dict) -> list[tuple[str, str, dict]]:
    """Return (run_name, registry_key, strategy_cfg) triples.

    When the 'rag' strategy has retrieval_mode as a list, it is expanded into one
    entry per mode (e.g. rag_dense, rag_bm25, rag_hybrid), each with an independent
    config dict so they run as separate tracked experiments.
    """
    expanded: list[tuple[str, str, dict]] = []
    for name in requested:
        if name == "rag":
            rag_cfg = config.get("rag", {})
            modes = rag_cfg.get("retrieval_mode", "hybrid")
            if isinstance(modes, list):
                for mode in modes:
                    expanded.append((f"rag_{mode}", "rag", {**rag_cfg, "retrieval_mode": mode}))
            else:
                expanded.append((name, name, rag_cfg))
        else:
            expanded.append((name, name, config.get(name, {})))
    return expanded


def run_mitigation(
    corrupted_dataset_path: str,
    baseline_results_path: str,
    config: dict,
    output_dir: str,
    strategies: list[str] | None = None,
) -> None:
    with open(corrupted_dataset_path) as f:
        dataset: list[dict] = json.load(f)

    max_samples = config.get("max_samples")
    subset_ids_path = config.get("subset_ids")
    if subset_ids_path:
        with open(subset_ids_path) as f:
            allowed = set(json.load(f))
        dataset = [item for item in dataset if item["sample_id"] in allowed]
        print(f"Filtered to {len(dataset)} samples from subset_ids file: {subset_ids_path}")
    elif max_samples is not None:
        dataset = dataset[:max_samples]

    # Per-model baselines recomputed on the same subset — fair delta comparison.
    sample_ids: set[str] = {item["sample_id"] for item in dataset}
    subset_baselines = load_subset_baselines(corrupted_dataset_path, sample_ids)
    _print_subset_baselines(subset_baselines, len(dataset))

    model_cfg = config.get("model", {})

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_baseline_subset(model_cfg, sample_ids, corrupted_dataset_path, out)

    # Fall back to the legacy single-file baseline if subset baselines are unavailable.
    baseline_metrics: dict = {}
    if Path(baseline_results_path).exists():
        with open(baseline_results_path) as f:
            baseline_data = json.load(f)
        baseline_metrics = baseline_data.get("metrics", {})

    requested = strategies or config.get("strategies", list(STRATEGIES.keys()))
    model_id = model_cfg.get("model_id", "unknown")
    mlflow.set_experiment(config.get("mlflow_experiment", "mitigation"))

    results: dict = {}

    prompt_strategy_entries = [
        (run_name, key, cfg)
        for run_name, key, cfg in _expand_strategies(requested, config)
        if key in STRATEGIES
    ]
    if prompt_strategy_entries:
        model = _load_model(model_cfg)
        for run_name, strategy_key, strategy_cfg in prompt_strategy_entries:
            strategy_cls = STRATEGIES[strategy_key]
            strategy = strategy_cls(strategy_cfg)
            strategy.name = run_name  # distinguish rag_dense / rag_bm25 / rag_hybrid in MLflow
            strategy.prepare(dataset, model)
            results[run_name] = evaluate_strategy(
                strategy=strategy,
                dataset=dataset,
                model=model,
                baseline_metrics=baseline_metrics,
                model_id=model_id,
                corrupted_dataset_path=corrupted_dataset_path,
                checkpoint_path=out / f"{run_name}_checkpoint.json",
            )

    if "finetuning" in requested:
        from .strategies.finetuning import FinetuningConfig, finetune
        ft_config = FinetuningConfig(**config.get("finetuning", {}))
        results["finetuning"] = {"metrics": finetune(dataset, ft_config)}

    subset_ids_out = out / "mitigation_subset_ids.json"
    with open(subset_ids_out, "w") as f:
        json.dump(sorted(sample_ids), f, indent=2)
    print(f"Subset IDs saved → {subset_ids_out}")

    # Attach subset baselines to the output so downstream analysis can compare fairly.
    output = {"subset_baselines": subset_baselines, **results}
    with open(out / "mitigation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved     → {out / 'mitigation_results.json'}")


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
    parser.add_argument("--corrupted_dataset", default=None)
    parser.add_argument("--baseline_results", default=None)
    parser.add_argument("--config", default="configs/mitigation_config.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--subset_ids",
        default=None,
        help="Path to a JSON file with a list of sample_id strings to run on (overrides max_samples)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Override strategies from config (e.g. --strategies rag few_shot)",
    )
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI args take precedence; fall back to config values.
    corrupted_dataset = args.corrupted_dataset or config.get("corrupted_dataset")
    if not corrupted_dataset:
        parser.error("corrupted_dataset must be set via --corrupted_dataset or in the config file")
    baseline_results = args.baseline_results or config.get("baseline_results", "")
    output_dir = args.output_dir or config.get("output_dir", "results/mitigation")
    if args.subset_ids:
        config["subset_ids"] = args.subset_ids

    run_mitigation(
        corrupted_dataset_path=corrupted_dataset,
        baseline_results_path=baseline_results,
        config=config,
        output_dir=output_dir,
        strategies=args.strategies,
    )


if __name__ == "__main__":
    main()
