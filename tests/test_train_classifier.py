import json

import torch

from src.benchmark.train_classifier import (
    build_embedding_cache,
    compute_cache_key,
    load_split,
    save_split,
    stratified_split,
)


def _make_records(n_unanswerable: int, n_answerable: int) -> list[dict]:
    records = []
    for i in range(n_unanswerable):
        records.append({"sample_id": f"u{i}", "is_unanswerable": True})
    for i in range(n_answerable):
        records.append({"sample_id": f"a{i}", "is_unanswerable": False})
    return records


def test_stratified_split_partitions_without_overlap():
    records = _make_records(100, 100)
    split = stratified_split(records, seed=42)

    train_ids = {r["sample_id"] for r in split["train"]}
    val_ids = {r["sample_id"] for r in split["val"]}
    test_ids = {r["sample_id"] for r in split["test"]}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == {r["sample_id"] for r in records}


def test_stratified_split_preserves_ratio_and_sizes():
    records = _make_records(100, 100)
    split = stratified_split(records, seed=42, ratios=(0.8, 0.1, 0.1))

    assert len(split["train"]) == 160
    assert len(split["val"]) == 20
    assert len(split["test"]) == 20

    for name in ("train", "val", "test"):
        n_unanswerable = sum(1 for r in split[name] if r["is_unanswerable"])
        ratio = n_unanswerable / len(split[name])
        assert abs(ratio - 0.5) < 0.05


def test_stratified_split_reproducible_with_same_seed():
    records = _make_records(50, 50)
    split_a = stratified_split(records, seed=7)
    split_b = stratified_split(records, seed=7)
    assert [r["sample_id"] for r in split_a["train"]] == [r["sample_id"] for r in split_b["train"]]


def test_save_and_load_split_roundtrip(tmp_path):
    records = _make_records(20, 20)
    split = stratified_split(records, seed=1)
    path = tmp_path / "split.json"

    save_split(split, str(path))
    assert json.loads(path.read_text())["train"]

    loaded = load_split(str(path), records)
    assert [r["sample_id"] for r in loaded["train"]] == [r["sample_id"] for r in split["train"]]
    assert [r["sample_id"] for r in loaded["test"]] == [r["sample_id"] for r in split["test"]]


class _FakeEncoders:
    def __init__(self):
        self.image_calls: list[str] = []
        self.window_calls: list[list[str]] = []
        self.text_calls: list[str] = []

    def encode_image(self, path: str) -> torch.Tensor:
        self.image_calls.append(path)
        return torch.ones(4) * len(path)

    def encode_image_window(self, paths: list[str]) -> torch.Tensor:
        self.window_calls.append(paths)
        return torch.stack([self.encode_image(p) for p in paths]).mean(dim=0)

    def encode_text(self, text: str) -> torch.Tensor:
        self.text_calls.append(text)
        return torch.ones(3) * len(text)


def test_compute_cache_key_changes_with_inputs():
    records = [{"sample_id": "a"}, {"sample_id": "b"}]
    key_a = compute_cache_key(records, "siglip-1", "minilm-1")
    key_b = compute_cache_key(records, "siglip-2", "minilm-1")
    key_c = compute_cache_key([{"sample_id": "a"}], "siglip-1", "minilm-1")

    assert key_a != key_b
    assert key_a != key_c
    assert key_a == compute_cache_key(records, "siglip-1", "minilm-1")


def test_build_embedding_cache_single_page(tmp_path):
    records = [
        {"sample_id": "s1", "document_path": "doc1.png", "question": "q1",
         "is_unanswerable": True, "metadata": {}},
    ]
    encoders = _FakeEncoders()
    cache_path = str(tmp_path / "cache.pt")
    key = compute_cache_key(records, "siglip-1", "minilm-1")

    cache = build_embedding_cache(records, encoders, cache_path, key)

    assert set(cache.keys()) == {"s1"}
    assert torch.equal(cache["s1"]["image_embed"], torch.ones(4) * len("doc1.png"))
    assert cache["s1"]["label"] is True
    assert encoders.image_calls == ["doc1.png"]
    assert encoders.window_calls == []


def test_build_embedding_cache_multi_page_window(tmp_path):
    records = [
        {"sample_id": "s1", "document_path": "docs/p1.png", "question": "q1",
         "is_unanswerable": False, "metadata": {"window_pages": ["p1", "p2"]}},
    ]
    encoders = _FakeEncoders()
    cache_path = str(tmp_path / "cache.pt")
    key = compute_cache_key(records, "siglip-1", "minilm-1")

    cache = build_embedding_cache(records, encoders, cache_path, key)

    assert encoders.window_calls == [["docs/p1.png", "docs/p2.png"]]


def test_build_embedding_cache_reuses_matching_cache(tmp_path):
    records = [
        {"sample_id": "s1", "document_path": "doc1.png", "question": "q1",
         "is_unanswerable": True, "metadata": {}},
    ]
    cache_path = str(tmp_path / "cache.pt")
    key = compute_cache_key(records, "siglip-1", "minilm-1")

    first_encoders = _FakeEncoders()
    build_embedding_cache(records, first_encoders, cache_path, key)

    second_encoders = _FakeEncoders()
    build_embedding_cache(records, second_encoders, cache_path, key)

    assert second_encoders.image_calls == []
    assert second_encoders.text_calls == []


def test_build_embedding_cache_rebuilds_on_key_mismatch(tmp_path):
    records = [
        {"sample_id": "s1", "document_path": "doc1.png", "question": "q1",
         "is_unanswerable": True, "metadata": {}},
    ]
    cache_path = str(tmp_path / "cache.pt")

    first_encoders = _FakeEncoders()
    build_embedding_cache(records, first_encoders, cache_path, compute_cache_key(records, "siglip-1", "minilm-1"))

    second_encoders = _FakeEncoders()
    build_embedding_cache(records, second_encoders, cache_path, compute_cache_key(records, "siglip-2", "minilm-1"))


class _FlakyEncoders(_FakeEncoders):
    def encode_image(self, path: str) -> torch.Tensor:
        if path == "bad.png":
            raise FileNotFoundError(path)
        return super().encode_image(path)


def test_build_embedding_cache_skips_record_on_encoder_error(tmp_path):
    records = [
        {"sample_id": "s1", "document_path": "doc1.png", "question": "q1",
         "is_unanswerable": True, "metadata": {}},
        {"sample_id": "bad", "document_path": "bad.png", "question": "q2",
         "is_unanswerable": False, "metadata": {}},
    ]
    encoders = _FlakyEncoders()
    cache_path = str(tmp_path / "cache.pt")
    key = compute_cache_key(records, "siglip-1", "minilm-1")

    cache = build_embedding_cache(records, encoders, cache_path, key)

    assert set(cache.keys()) == {"s1"}


def test_build_embedding_cache_resumes_partial_progress(tmp_path):
    records = [
        {"sample_id": "s1", "document_path": "doc1.png", "question": "q1",
         "is_unanswerable": True, "metadata": {}},
        {"sample_id": "s2", "document_path": "doc2.png", "question": "q2",
         "is_unanswerable": False, "metadata": {}},
    ]
    cache_path = str(tmp_path / "cache.pt")
    key = compute_cache_key(records, "siglip-1", "minilm-1")

    # Pre-seed the cache file with only the first record already encoded,
    # simulating an interrupted run that saved incrementally.
    torch.save(
        {"cache_key": key, "embeddings": {
            "s1": {"image_embed": torch.ones(4), "text_embed": torch.ones(3), "label": True},
        }},
        cache_path,
    )

    encoders = _FakeEncoders()
    cache = build_embedding_cache(records, encoders, cache_path, key)

    assert set(cache.keys()) == {"s1", "s2"}
    assert encoders.image_calls == ["doc2.png"]


from src.benchmark.train_classifier import evaluate_head, train_head
from src.benchmark.models.siglip_classifier import ClassifierHead, IMAGE_EMBED_DIM, TEXT_EMBED_DIM


def _make_linearly_separable_embeddings(n_per_class: int) -> tuple[list[dict], dict[str, dict]]:
    records = []
    embeddings = {}
    for i in range(n_per_class):
        sid = f"pos{i}"
        records.append({"sample_id": sid, "is_unanswerable": True})
        embeddings[sid] = {
            "image_embed": torch.ones(IMAGE_EMBED_DIM) * 5.0,
            "text_embed": torch.ones(TEXT_EMBED_DIM) * 5.0,
            "label": True,
        }
        sid = f"neg{i}"
        records.append({"sample_id": sid, "is_unanswerable": False})
        embeddings[sid] = {
            "image_embed": torch.ones(IMAGE_EMBED_DIM) * -5.0,
            "text_embed": torch.ones(TEXT_EMBED_DIM) * -5.0,
            "label": False,
        }
    return records, embeddings


def test_train_head_separates_linearly_separable_data():
    records, embeddings = _make_linearly_separable_embeddings(40)
    train_records = records[:64]
    val_records = records[64:]

    head = train_head(train_records, val_records, embeddings, epochs=30, lr=1e-2, batch_size=16)
    metrics = evaluate_head(head, val_records, embeddings)

    assert metrics.f1 > 0.9


def test_evaluate_head_returns_benchmark_metrics():
    records, embeddings = _make_linearly_separable_embeddings(10)
    head = ClassifierHead()
    metrics = evaluate_head(head, records, embeddings)

    assert hasattr(metrics, "f1")
    assert hasattr(metrics, "mcc")
