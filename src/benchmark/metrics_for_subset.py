"""Compute benchmark metrics restricted to a subset of sample ids (e.g. a train/val/test split).

Usage:
    uv run python -m src.benchmark.metrics_for_subset \\
        --results "results/benchmark_docvqa/*_benchmark_result.json" \\
        --split data/cache/siglip_classifier_split.json \\
        --split_name test \\
        --output results/benchmark_docvqa/test_metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .evaluation.metrics import compute_metrics, compute_per_type_metrics


def _record_key(record: dict) -> str:
    """Matches the key format used in data/cache/siglip_classifier_split.json
    (see src/benchmark/train_classifier.py:_record_key)."""
    return f"{record['sample_id']}::{int(bool(record['label_unanswerable']))}"


def load_subset_keys(split_path: str, split_name: str) -> set[str]:
    split = json.loads(Path(split_path).read_text())
    if split_name not in split:
        raise KeyError(f"split_name {split_name!r} not in {split_path} (available: {list(split)})")
    return set(split[split_name])


def filter_records(records: list[dict], keys: set[str]) -> list[dict]:
    return [r for r in records if _record_key(r) in keys]


def _mean(records: list[dict], field: str) -> float | None:
    values = [r[field] for r in records if field in r]
    return sum(values) / len(values) if values else None


def metrics_for_result_file(path: str, keys: "set[str] | None") -> dict:
    data = json.loads(Path(path).read_text())
    records = data["records"]
    if keys is not None:
        records = filter_records(records, keys)
    records = [r for r in records if not r.get("skipped", False)]

    y_true = [r["label_unanswerable"] for r in records]
    y_pred = [r["predicted_unanswerable"] for r in records]
    overall = compute_metrics(y_true, y_pred)
    per_type = compute_per_type_metrics(records)
    return {
        "n": len(records),
        "overall": overall,
        "per_type": per_type,
        "mean_inference_time_s": _mean(records, "inference_time_s"),
        "mean_response_length": _mean(records, "response_length"),
    }


def _to_jsonable(result: dict) -> dict:
    return {
        "n": result["n"],
        "overall": result["overall"].__dict__,
        "per_type": {ctype: m.__dict__ for ctype, m in result["per_type"].items()},
        "mean_inference_time_s": result["mean_inference_time_s"],
        "mean_response_length": result["mean_response_length"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True,
                         help="Path(s) or glob pattern(s) to *_benchmark_result.json files")
    parser.add_argument("--split", default=None,
                         help="Path to a split JSON ({'train': [...], 'val': [...], ...}); "
                              "omit to evaluate on all records in each results file")
    parser.add_argument("--split_name", default="val",
                         help="Key into the split file to use as the id subset")
    parser.add_argument("--output", default=None,
                         help="Path to write a JSON report (keyed by model name); omit to only print")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.results:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])

    keys = load_subset_keys(args.split, args.split_name) if args.split else None

    report: dict = {}
    for path in paths:
        model_name = Path(path).stem.removesuffix("_benchmark_result")
        result = metrics_for_result_file(path, keys)
        report[model_name] = _to_jsonable(result)
        print(f"\n=== {model_name} (n={result['n']}) ===")
        print(f"  Overall: {result['overall']}")
        for ctype, m in sorted(result["per_type"].items()):
            print(f"  [{ctype}]: {m}")
        if result["mean_inference_time_s"] is not None:
            print(f"  Mean inference_time_s: {result['mean_inference_time_s']:.6f}")
        if result["mean_response_length"] is not None:
            print(f"  Mean response_length: {result['mean_response_length']:.2f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"\nReport saved → {args.output}")


if __name__ == "__main__":
    main()
