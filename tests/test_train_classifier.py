import json

from src.benchmark.train_classifier import load_split, save_split, stratified_split


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
