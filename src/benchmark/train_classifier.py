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
from tqdm import tqdm

from .evaluation.metrics import BenchmarkMetrics, compute_metrics, plot_confusion_matrix
from .models.siglip_classifier import ClassifierHead, PretrainedEncoders


def _record_key(record: dict) -> str:
    """Unique key for a benchmark record. sample_id alone is not unique: the original
    and corrupted half of a pair share the same sample_id but differ in is_unanswerable."""
    return f"{record['sample_id']}::{int(bool(record['is_unanswerable']))}"


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
    keys = {name: [_record_key(r) for r in records] for name, records in split.items()}
    with open(path, "w") as f:
        json.dump(keys, f, indent=2)


def load_split(path: str, records: list[dict]) -> dict[str, list[dict]]:
    with open(path) as f:
        keys = json.load(f)
    by_key = {_record_key(r): r for r in records}
    return {name: [by_key[key] for key in record_keys] for name, record_keys in keys.items()}


def compute_cache_key(records: list[dict], siglip_model_id: str, minilm_model_id: str) -> str:
    """Compute a stable hash key from sorted record keys and model ids."""
    record_keys = sorted(_record_key(r) for r in records)
    payload = json.dumps([record_keys, siglip_model_id, minilm_model_id], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_embedding_cache(
    records: list[dict],
    encoders,
    cache_path: str,
    cache_key: str,
    batch_size: int = 32,
) -> dict[str, dict]:
    """Build or load a cached embedding dictionary, invalidating on key mismatch.

    Single-page document images and question texts are deduplicated and batch-encoded
    (many records share the same document image or question string), instead of running
    one image/text through the model at a time.

    Args:
        records: List of benchmark records with sample_id, document_path, question, is_unanswerable, metadata.
        encoders: Object with encode_image(path), encode_text(text), optionally encode_images(paths),
            encode_texts(texts), and encode_image_window(paths).
        cache_path: Path to save/load the cache file.
        cache_key: Expected cache key; if file exists but key doesn't match, rebuilds.
        batch_size: Number of unique images encoded per forward pass.

    Returns:
        dict[str, dict]: {record_key: {"image_embed": Tensor, "text_embed": Tensor, "label": bool}}
        where record_key is _record_key(record), since sample_id alone is not unique.
    """
    path = Path(cache_path)
    embeddings: dict[str, dict] = {}
    if path.exists():
        stored = torch.load(path, map_location="cpu")
        if stored.get("cache_key") == cache_key:
            embeddings = stored["embeddings"]
            if all(_record_key(r) in embeddings for r in records):
                return embeddings

    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [r for r in records if _record_key(r) not in embeddings]
    if not missing:
        torch.save({"cache_key": cache_key, "embeddings": embeddings}, path)
        return embeddings

    use_window = hasattr(encoders, "encode_image_window")

    def _is_window_record(record: dict) -> bool:
        return use_window and bool(record.get("metadata", {}).get("window_pages"))

    # Dedup + batch encode single-page document images shared across many records.
    single_page_paths = sorted({r["document_path"] for r in missing if not _is_window_record(r)})
    image_embed_by_path: dict[str, torch.Tensor] = {}
    bad_paths: set[str] = set()
    for start in tqdm(range(0, len(single_page_paths), batch_size), desc="Encoding images", unit="batch"):
        chunk = single_page_paths[start:start + batch_size]
        try:
            if hasattr(encoders, "encode_images"):
                chunk_embeds = encoders.encode_images(chunk)
            else:
                chunk_embeds = torch.stack([encoders.encode_image(p) for p in chunk])
        except Exception:
            for p in chunk:
                try:
                    image_embed_by_path[p] = encoders.encode_image(p)
                except Exception:
                    bad_paths.add(p)
            continue
        for p, embed in zip(chunk, chunk_embeds):
            image_embed_by_path[p] = embed

    # Dedup + batch encode all question texts (single- and multi-page records alike).
    unique_questions = sorted({r["question"] for r in missing})
    bad_questions: set[str] = set()
    try:
        if hasattr(encoders, "encode_texts"):
            text_embeds = encoders.encode_texts(unique_questions)
            text_embed_by_question = dict(zip(unique_questions, text_embeds))
        else:
            text_embed_by_question = {q: encoders.encode_text(q) for q in unique_questions}
    except Exception:
        text_embed_by_question = {}
        for q in unique_questions:
            try:
                text_embed_by_question[q] = encoders.encode_text(q)
            except Exception:
                bad_questions.add(q)

    save_every = 200
    progress = tqdm(missing, desc="Assembling embeddings", unit="sample")
    for i, record in enumerate(progress):
        key = _record_key(record)
        try:
            if record["question"] in bad_questions:
                raise RuntimeError(f"failed to encode question for {record['sample_id']}")
            if _is_window_record(record):
                document_dir = Path(record["document_path"]).parent
                suffix = Path(record["document_path"]).suffix
                window_pages = record["metadata"]["window_pages"]
                window_paths = [str(document_dir / f"{page_id}{suffix}") for page_id in window_pages]
                image_embed = encoders.encode_image_window(window_paths)
            else:
                if record["document_path"] in bad_paths:
                    raise RuntimeError(f"failed to encode image {record['document_path']}")
                image_embed = image_embed_by_path[record["document_path"]]
            text_embed = text_embed_by_question[record["question"]]
        except Exception as exc:
            progress.write(f"Skipping {record['sample_id']}: {exc}")
            continue
        embeddings[key] = {
            "image_embed": image_embed.detach().cpu(),
            "text_embed": text_embed.detach().cpu(),
            "label": bool(record["is_unanswerable"]),
        }
        if (i + 1) % save_every == 0:
            torch.save({"cache_key": cache_key, "embeddings": embeddings}, path)

    torch.save({"cache_key": cache_key, "embeddings": embeddings}, path)
    return embeddings


def _batch_tensors(records: list[dict], embeddings: dict[str, dict]):
    image_embeds = torch.stack([embeddings[_record_key(r)]["image_embed"] for r in records])
    text_embeds = torch.stack([embeddings[_record_key(r)]["text_embed"] for r in records])
    labels = torch.tensor([float(embeddings[_record_key(r)]["label"]) for r in records])
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

    epoch_bar = tqdm(range(epochs), desc="Training head", unit="epoch")
    for _epoch in epoch_bar:
        head.train()
        order = list(range(len(train_records)))
        random.Random(_epoch).shuffle(order)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            batch_records = [train_records[i] for i in batch_idx]
            image_embeds, text_embeds, labels = _batch_tensors(batch_records, embeddings)
            optimizer.zero_grad()
            logits = head(image_embeds, text_embeds)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        val_metrics = evaluate_head(head, val_records, embeddings)
        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
            best_state = copy.deepcopy(head.state_dict())
        epoch_bar.set_postfix(loss=epoch_loss / num_batches, val_f1=val_metrics.f1, best_f1=best_f1)

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
        split_ratios = tuple(config.get("split_ratios", (0.8, 0.1, 0.1)))
        split = stratified_split(records, seed=config.get("seed", 42), ratios=split_ratios)
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
