"""Prepare a mixed benchmark dataset from a corrupted JSON.

Each corrupted record contributes two benchmark samples:
  - positive: corrupted_question  → is_unanswerable=True
  - negative: original_question   → is_unanswerable=False

Output is written to data/benchmark/{dataset}_benchmark.json.

Usage:
    uv run prepare-benchmark \
        --corrupted_dataset data/corrupted/docvqa_corrupted.json \
        --output_dir data/benchmark
"""

import argparse
import json
import random
from pathlib import Path


def prepare_benchmark(
    corrupted_dataset_path: str,
    output_dir: str,
    seed: int = 42,
) -> list[dict]:
    with open(corrupted_dataset_path) as f:
        records = json.load(f)

    rng = random.Random(seed)
    mixed: list[dict] = []
    for rec in records:
        base = {
            "sample_id": rec["sample_id"],
            "document_path": rec["document_path"],
            "original_answer": rec["original_answer"],
            "page_index": rec["page_index"],
            "metadata": rec.get("metadata", {}),
            "corruption_type": rec["corruption_type"],
            "corruption_detail": rec.get("corruption_detail", ""),
        }
        mixed.append({**base, "question": rec["corrupted_question"], "is_unanswerable": True})
        mixed.append({**base, "question": rec["original_question"], "is_unanswerable": False})

    rng.shuffle(mixed)

    dataset_name = Path(corrupted_dataset_path).stem.replace("_corrupted", "")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{dataset_name}_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(mixed, f, indent=2)

    pos = sum(1 for r in mixed if r["is_unanswerable"])
    print(f"Wrote {len(mixed)} samples ({pos} unanswerable, {len(mixed) - pos} answerable) → {out_path}")
    return mixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted_dataset", required=True)
    parser.add_argument("--output_dir", default="data/benchmark")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_benchmark(args.corrupted_dataset, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
