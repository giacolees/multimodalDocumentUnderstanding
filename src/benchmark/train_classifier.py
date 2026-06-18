"""Offline training script for the SigLIP classifier backend's MLP head."""

from __future__ import annotations

import json
import random
from collections import defaultdict


def stratified_split(
    records: list[dict],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[dict]]:
    """Split records into train/val/test, preserving the is_unanswerable ratio in each split."""
    buckets: dict[bool, list[dict]] = defaultdict(list)
    for r in records:
        buckets[bool(r["is_unanswerable"])].append(r)

    rng = random.Random(seed)
    split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    train_ratio, val_ratio, _test_ratio = ratios
    for bucket_records in buckets.values():
        shuffled = bucket_records[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        split["train"].extend(shuffled[:n_train])
        split["val"].extend(shuffled[n_train:n_train + n_val])
        split["test"].extend(shuffled[n_train + n_val:])

    for name in split:
        rng.shuffle(split[name])
    return split


def save_split(split: dict[str, list[dict]], path: str) -> None:
    ids = {name: [r["sample_id"] for r in records] for name, records in split.items()}
    with open(path, "w") as f:
        json.dump(ids, f, indent=2)


def load_split(path: str, records: list[dict]) -> dict[str, list[dict]]:
    with open(path) as f:
        ids = json.load(f)
    by_id = {r["sample_id"]: r for r in records}
    return {name: [by_id[sid] for sid in sample_ids] for name, sample_ids in ids.items()}
