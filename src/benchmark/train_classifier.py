"""Offline training script for the SigLIP classifier backend's MLP head."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from .evaluation.metrics import BenchmarkMetrics, compute_metrics, plot_confusion_matrix
from .models.siglip_classifier import ClassifierHead, PretrainedEncoders


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


def _batch_tensors(records: list[dict], embeddings: dict[str, dict]):
    image_embeds = torch.stack([embeddings[r["sample_id"]]["image_embed"] for r in records])
    text_embeds = torch.stack([embeddings[r["sample_id"]]["text_embed"] for r in records])
    labels = torch.tensor([float(embeddings[r["sample_id"]]["label"]) for r in records])
    return image_embeds, text_embeds, labels


def train_head(
    train_records: list[dict],
    val_records: list[dict],
    embeddings: dict[str, dict],
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> ClassifierHead:
    head = ClassifierHead()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_state = copy.deepcopy(head.state_dict())

    for _epoch in range(epochs):
        head.train()
        order = list(range(len(train_records)))
        random.Random(_epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            batch_records = [train_records[i] for i in batch_idx]
            image_embeds, text_embeds, labels = _batch_tensors(batch_records, embeddings)
            optimizer.zero_grad()
            logits = head(image_embeds, text_embeds)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_head(head, val_records, embeddings)
        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
            best_state = copy.deepcopy(head.state_dict())

    best_head = ClassifierHead()
    best_head.load_state_dict(best_state)
    return best_head


def evaluate_head(
    head: ClassifierHead,
    records: list[dict],
    embeddings: dict[str, dict],
) -> BenchmarkMetrics:
    head.eval()
    image_embeds, text_embeds, labels = _batch_tensors(records, embeddings)
    with torch.no_grad():
        probs = torch.sigmoid(head(image_embeds, text_embeds))
    preds = (probs >= 0.5).tolist()
    y_true = [bool(v) for v in labels.tolist()]
    return compute_metrics(y_true, preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    records: list[dict] = []
    for path in config["dataset_paths"]:
        with open(path) as f:
            records.extend(json.load(f))

    split_path = config["split_path"]
    if Path(split_path).exists():
        split = load_split(split_path, records)
    else:
        split = stratified_split(records, seed=config.get("seed", 42))
        save_split(split, split_path)

    encoders = PretrainedEncoders(
        siglip_model_id=config["siglip_model_id"],
        minilm_model_id=config["minilm_model_id"],
        device=config.get("device", "cpu"),
    )
    cache_key = compute_cache_key(records, config["siglip_model_id"], config["minilm_model_id"])
    embeddings = build_embedding_cache(records, encoders, config["cache_path"], cache_key)

    import mlflow
    mlflow.set_experiment("siglip_classifier")
    with mlflow.start_run():
        mlflow.log_params({
            "siglip_model_id": config["siglip_model_id"],
            "minilm_model_id": config["minilm_model_id"],
            "epochs": config.get("epochs", 20),
            "lr": config.get("lr", 1e-3),
            "batch_size": config.get("batch_size", 64),
        })
        head = train_head(
            split["train"], split["val"], embeddings,
            epochs=config.get("epochs", 20),
            lr=config.get("lr", 1e-3),
            batch_size=config.get("batch_size", 64),
        )
        test_metrics = evaluate_head(head, split["test"], embeddings)
        mlflow.log_metrics({
            "accuracy": test_metrics.accuracy,
            "precision": test_metrics.precision,
            "recall": test_metrics.recall,
            "f1": test_metrics.f1,
            "mcc": test_metrics.mcc,
        })
        fig = plot_confusion_matrix(test_metrics, title="SigLIP Classifier — test set")
        mlflow.log_figure(fig, "confusion_matrix_siglip_classifier.png")
        plt.close(fig)

    head_path = Path(config["head_checkpoint_path"])
    head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), head_path)
    print(f"Saved head to {head_path}")
    print(test_metrics)


if __name__ == "__main__":
    main()
