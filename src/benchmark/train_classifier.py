"""Offline training script for the SigLIP classifier backend's MLP head."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch


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


def compute_cache_key(records: list[dict], siglip_model_id: str, minilm_model_id: str) -> str:
    """Compute a stable hash key from sorted sample ids and model ids."""
    sample_ids = sorted(r["sample_id"] for r in records)
    payload = json.dumps([sample_ids, siglip_model_id, minilm_model_id], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_embedding_cache(
    records: list[dict],
    encoders,
    cache_path: str,
    cache_key: str,
) -> dict[str, dict]:
    """Build or load a cached embedding dictionary, invalidating on key mismatch.

    Args:
        records: List of benchmark records with sample_id, document_path, question, is_unanswerable, metadata.
        encoders: Object with encode_image(path), encode_text(text), and optionally encode_image_window(paths).
        cache_path: Path to save/load the cache file.
        cache_key: Expected cache key; if file exists but key doesn't match, rebuilds.

    Returns:
        dict[str, dict]: {sample_id: {"image_embed": Tensor, "text_embed": Tensor, "label": bool}}
    """
    path = Path(cache_path)
    if path.exists():
        stored = torch.load(path, map_location="cpu")
        if stored.get("cache_key") == cache_key:
            return stored["embeddings"]

    embeddings: dict[str, dict] = {}
    for record in records:
        window_pages = record.get("metadata", {}).get("window_pages") or []
        if window_pages and hasattr(encoders, "encode_image_window"):
            image_embed = encoders.encode_image_window(window_pages)
        else:
            image_embed = encoders.encode_image(record["document_path"])
        text_embed = encoders.encode_text(record["question"])
        embeddings[record["sample_id"]] = {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "label": bool(record["is_unanswerable"]),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cache_key": cache_key, "embeddings": embeddings}, path)
    return embeddings
