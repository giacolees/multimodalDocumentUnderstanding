# SigLIP Classifier Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fast, pure vision-transformer baseline (frozen SigLIP image encoder + frozen MiniLM text encoder + small trained MLP head) for unanswerable-question detection, integrated as a normal `BaseVisionModel` backend so it's directly comparable to the existing VLM benchmark results.

**Architecture:** `ClassifierHead` (trainable MLP) consumes a concatenated [SigLIP image embedding ‖ MiniLM text embedding] vector and outputs a single logit. `SiglipClassifierModel` wraps a `ClassifierHead` plus an encoder object behind the `BaseVisionModel` interface for benchmark-time inference (single page per sample, matching how every other backend already works). A separate offline script, `train_classifier.py`, builds a combined docvqa+mp_docvqa dataset, does a stratified split, builds a cached embedding table (mean-pooling over `metadata.window_pages` for multi-page mp_docvqa samples), trains the head, and evaluates it with the existing metrics module.

**Tech Stack:** PyTorch, HuggingFace `transformers` (SigLIP), `sentence-transformers` (MiniLM), existing `src/benchmark/evaluation/metrics.py`, MLflow, pytest + `unittest.mock`.

## Global Constraints

- Reuse `compute_metrics`, `compute_per_type_metrics`, `plot_confusion_matrix` from `src/benchmark/evaluation/metrics.py` — no duplicate metrics code.
- `SiglipClassifierModel` must implement `BaseVisionModel` (`predict_unanswerable()`, `name()`) exactly as defined in `src/benchmark/models/base_model.py`.
- At benchmark-inference time, only `document_path` is available (no `metadata.window_pages`) — this matches every existing backend (`VllmModel` etc. also only look at one page), so `SiglipClassifierModel.predict_unanswerable()` only encodes the single page at `document_path`.
- Multi-page mean-pooling over `metadata.window_pages` only happens inside `train_classifier.py`, which has direct access to the full dataset records.
- New runtime dependency group: `vit-classifier = ["torch>=2.2", "transformers>=4.40", "scikit-learn>=1.4"]` in `pyproject.toml`. `sentence-transformers` is reused from the existing `mitigation` extra.
- Results must land in `results/benchmark_{dataset}/siglip_classifier_benchmark_result.json`, same shape as other backends (handled automatically by `run_benchmark.py` once the backend is registered).
- Tests follow this repo's existing style: plain `pytest` functions (no fixture frameworks), `unittest.mock.Mock`/`mock.patch` for anything that would otherwise hit the network or load a real model (see `tests/test_mlflow_tracking.py`, `tests/test_generate.py`).

---

### Task 1: Add the `vit-classifier` dependency group

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `torch`, `transformers`, `scikit-learn` importable after `uv sync --extra vit-classifier --extra mitigation`.

- [ ] **Step 1: Add the optional dependency group**

In `pyproject.toml`, after the existing `mitigation = [...]` block, add:

```toml
vit-classifier = [
    "torch>=2.2",
    "transformers>=4.40",
    "scikit-learn>=1.4",
]
```

- [ ] **Step 2: Sync and verify imports**

Run: `uv sync --extra vit-classifier --extra mitigation`
Then run: `uv run python -c "import torch, transformers, sentence_transformers, sklearn; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add vit-classifier optional dependency group"
```

---

### Task 2: `ClassifierHead` MLP module

**Files:**
- Create: `src/benchmark/models/siglip_classifier.py`
- Test: `tests/test_siglip_classifier.py`

**Interfaces:**
- Produces: `ClassifierHead(image_dim: int = 1152, text_dim: int = 384, hidden_dims: tuple[int, int] = (512, 128))`, a `torch.nn.Module` whose `forward(image_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor` takes batched `(batch, image_dim)` and `(batch, text_dim)` tensors and returns raw logits of shape `(batch,)`.
- Produces: module-level constants `IMAGE_EMBED_DIM = 1152`, `TEXT_EMBED_DIM = 384`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_siglip_classifier.py`:

```python
import torch

from src.benchmark.models.siglip_classifier import (
    ClassifierHead,
    IMAGE_EMBED_DIM,
    TEXT_EMBED_DIM,
)


def test_classifier_head_forward_shape():
    head = ClassifierHead()
    image_embed = torch.randn(4, IMAGE_EMBED_DIM)
    text_embed = torch.randn(4, TEXT_EMBED_DIM)
    logits = head(image_embed, text_embed)
    assert logits.shape == (4,)
    assert logits.dtype == torch.float32


def test_classifier_head_gradients_flow():
    head = ClassifierHead()
    image_embed = torch.randn(2, IMAGE_EMBED_DIM)
    text_embed = torch.randn(2, TEXT_EMBED_DIM)
    labels = torch.tensor([1.0, 0.0])
    logits = head(image_embed, text_embed)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    grads = [p.grad for p in head.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_siglip_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.benchmark.models.siglip_classifier'`

- [ ] **Step 3: Implement `ClassifierHead`**

Create `src/benchmark/models/siglip_classifier.py`:

```python
"""SigLIP image encoder + MiniLM text encoder + trained MLP head classifier backend."""

from __future__ import annotations

import time
from typing import Protocol

import torch
from torch import nn

from .base_model import BaseVisionModel, PredictionResult

IMAGE_EMBED_DIM = 1152
TEXT_EMBED_DIM = 384


class ClassifierHead(nn.Module):
    """Trainable MLP fusing a frozen image embedding and a frozen text embedding."""

    def __init__(
        self,
        image_dim: int = IMAGE_EMBED_DIM,
        text_dim: int = TEXT_EMBED_DIM,
        hidden_dims: tuple[int, int] = (512, 128),
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(image_dim + text_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, image_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([image_embed, text_embed], dim=-1)
        return self.net(fused).squeeze(-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_siglip_classifier.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/models/siglip_classifier.py tests/test_siglip_classifier.py
git commit -m "feat: add ClassifierHead MLP for SigLIP classifier backend"
```

---

### Task 3: `SiglipClassifierModel` (BaseVisionModel) with injectable encoders

**Files:**
- Modify: `src/benchmark/models/siglip_classifier.py`
- Test: `tests/test_siglip_classifier.py`

**Interfaces:**
- Consumes: `ClassifierHead` from Task 2; `BaseVisionModel`, `PredictionResult` from `src/benchmark/models/base_model.py`.
- Produces: `ImageTextEncoder` Protocol with `encode_image(self, image_path: str) -> torch.Tensor` and `encode_text(self, text: str) -> torch.Tensor` (each returning a 1-D tensor of dim `IMAGE_EMBED_DIM`/`TEXT_EMBED_DIM`).
- Produces: `SiglipClassifierModel(encoders: ImageTextEncoder, head: ClassifierHead, model_id: str = "siglip_classifier", threshold: float = 0.5)` implementing `predict_unanswerable(document_path, question, prompt_template, page_indices=None) -> PredictionResult` and `name() -> str`.
- Produces: `PretrainedEncoders(siglip_model_id: str, minilm_model_id: str, device: str = "cpu")` implementing the `ImageTextEncoder` protocol with real models, plus `encode_image_window(self, image_paths: list[str]) -> torch.Tensor` (mean-pools `encode_image` over multiple paths) — used by `train_classifier.py` in Task 7, not by `predict_unanswerable`.
- Produces: `SiglipClassifierModel.from_pretrained(head_checkpoint_path: str, siglip_model_id: str = "google/siglip-so400m-patch14-384", minilm_model_id: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu") -> SiglipClassifierModel` classmethod that loads real encoders + head weights from disk.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_siglip_classifier.py`:

```python
from unittest import mock

from src.benchmark.models.base_model import PredictionResult
from src.benchmark.models.siglip_classifier import SiglipClassifierModel


class _FakeEncoders:
    def __init__(self, image_value: float, text_value: float):
        self.image_value = image_value
        self.text_value = text_value
        self.image_calls: list[str] = []
        self.text_calls: list[str] = []

    def encode_image(self, image_path: str) -> torch.Tensor:
        self.image_calls.append(image_path)
        return torch.full((IMAGE_EMBED_DIM,), self.image_value)

    def encode_text(self, text: str) -> torch.Tensor:
        self.text_calls.append(text)
        return torch.full((TEXT_EMBED_DIM,), self.text_value)


def _head_that_always_outputs(logit_value: float) -> ClassifierHead:
    head = ClassifierHead()
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net[-1].bias.fill_(logit_value)
    return head


def test_predict_unanswerable_above_threshold():
    encoders = _FakeEncoders(image_value=1.0, text_value=1.0)
    head = _head_that_always_outputs(10.0)  # sigmoid(10) ~ 0.99995
    model = SiglipClassifierModel(encoders=encoders, head=head)

    result = model.predict_unanswerable(
        document_path="data/raw/docvqa/val/documents/416.png",
        question="What is the full form of FDA?",
        prompt_template="ignored",
    )

    assert isinstance(result, PredictionResult)
    assert result.predicted_unanswerable is True
    assert result.confidence > 0.99
    assert result.skipped is False
    assert encoders.image_calls == ["data/raw/docvqa/val/documents/416.png"]
    assert encoders.text_calls == ["What is the full form of FDA?"]


def test_predict_unanswerable_below_threshold():
    encoders = _FakeEncoders(image_value=1.0, text_value=1.0)
    head = _head_that_always_outputs(-10.0)  # sigmoid(-10) ~ 0.00005
    model = SiglipClassifierModel(encoders=encoders, head=head)

    result = model.predict_unanswerable(
        document_path="doc.png", question="q", prompt_template="ignored",
    )

    assert result.predicted_unanswerable is False
    assert result.confidence < 0.01


def test_predict_unanswerable_handles_encoder_failure():
    encoders = mock.Mock()
    encoders.encode_image.side_effect = FileNotFoundError("missing image")
    head = ClassifierHead()
    model = SiglipClassifierModel(encoders=encoders, head=head)

    result = model.predict_unanswerable(
        document_path="missing.png", question="q", prompt_template="ignored",
    )

    assert result.skipped is True
    assert result.predicted_unanswerable is False
    assert "missing image" in result.raw_response


def test_name_returns_model_id():
    model = SiglipClassifierModel(
        encoders=mock.Mock(), head=ClassifierHead(), model_id="siglip_classifier",
    )
    assert model.name() == "siglip_classifier"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_siglip_classifier.py -v`
Expected: FAIL — `ImportError: cannot import name 'SiglipClassifierModel'`

- [ ] **Step 3: Implement `ImageTextEncoder`, `SiglipClassifierModel`, `PretrainedEncoders`**

Add to `src/benchmark/models/siglip_classifier.py` (after `ClassifierHead`):

```python
class ImageTextEncoder(Protocol):
    def encode_image(self, image_path: str) -> torch.Tensor: ...
    def encode_text(self, text: str) -> torch.Tensor: ...


class SiglipClassifierModel(BaseVisionModel):
    """BaseVisionModel backend: frozen image/text encoders + trained MLP head."""

    def __init__(
        self,
        encoders: ImageTextEncoder,
        head: ClassifierHead,
        model_id: str = "siglip_classifier",
        threshold: float = 0.5,
    ) -> None:
        self._encoders = encoders
        self._head = head
        self._head.eval()
        self._model_id = model_id
        self._threshold = threshold

    def name(self) -> str:
        return self._model_id

    def predict_unanswerable(
        self,
        document_path: str,
        question: str,
        prompt_template: str,
        page_indices: list[int] | None = None,
    ) -> PredictionResult:
        t0 = time.perf_counter()
        try:
            image_embed = self._encoders.encode_image(document_path)
            text_embed = self._encoders.encode_text(question)
            with torch.no_grad():
                logit = self._head(image_embed.unsqueeze(0), text_embed.unsqueeze(0))
                prob = torch.sigmoid(logit).item()
        except Exception as exc:
            return PredictionResult(
                sample_id="",
                predicted_unanswerable=False,
                confidence=-1.0,
                raw_response=f"error: {exc}",
                inference_time_s=time.perf_counter() - t0,
                skipped=True,
            )
        return PredictionResult(
            sample_id="",
            predicted_unanswerable=prob >= self._threshold,
            confidence=prob,
            raw_response=f"p_unanswerable={prob:.4f}",
            inference_time_s=time.perf_counter() - t0,
            skipped=False,
        )

    @classmethod
    def from_pretrained(
        cls,
        head_checkpoint_path: str,
        siglip_model_id: str = "google/siglip-so400m-patch14-384",
        minilm_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> "SiglipClassifierModel":
        encoders = PretrainedEncoders(
            siglip_model_id=siglip_model_id, minilm_model_id=minilm_model_id, device=device,
        )
        head = ClassifierHead()
        state = torch.load(head_checkpoint_path, map_location=device)
        head.load_state_dict(state)
        head.to(device)
        return cls(encoders=encoders, head=head)


class PretrainedEncoders:
    """Real frozen SigLIP + MiniLM encoders. Loads actual model weights — not unit-tested."""

    def __init__(self, siglip_model_id: str, minilm_model_id: str, device: str = "cpu") -> None:
        from PIL import Image
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModel, AutoProcessor

        self._Image = Image
        self._device = device
        self._siglip = AutoModel.from_pretrained(siglip_model_id).to(device).eval()
        self._processor = AutoProcessor.from_pretrained(siglip_model_id)
        self._minilm = SentenceTransformer(minilm_model_id, device=device)

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        image = self._Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        return self._siglip.get_image_features(**inputs).squeeze(0)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        embedding = self._minilm.encode(text, convert_to_tensor=True)
        return embedding.to(self._device)

    def encode_image_window(self, image_paths: list[str]) -> torch.Tensor:
        embeds = torch.stack([self.encode_image(p) for p in image_paths])
        return embeds.mean(dim=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_siglip_classifier.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/models/siglip_classifier.py tests/test_siglip_classifier.py
git commit -m "feat: add SiglipClassifierModel BaseVisionModel backend"
```

---

### Task 4: Register `siglip_classifier` backend in `run_benchmark.py`

**Files:**
- Modify: `src/benchmark/run_benchmark.py:41-73` (`load_model`)
- Test: `tests/test_run_benchmark_siglip_backend.py`

**Interfaces:**
- Consumes: `SiglipClassifierModel.from_pretrained(...)` from Task 3.
- Produces: `load_model({"backend": "siglip_classifier", ...})` returns a `SiglipClassifierModel`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_benchmark_siglip_backend.py`:

```python
from unittest import mock

from src.benchmark.run_benchmark import load_model


def test_load_model_siglip_classifier_backend():
    fake_model = mock.Mock()
    with mock.patch(
        "src.benchmark.models.siglip_classifier.SiglipClassifierModel.from_pretrained",
        return_value=fake_model,
    ) as mock_from_pretrained:
        result = load_model({
            "backend": "siglip_classifier",
            "head_checkpoint_path": "models/siglip_classifier_head.pt",
            "siglip_model_id": "google/siglip-so400m-patch14-384",
            "minilm_model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
        })

    assert result is fake_model
    mock_from_pretrained.assert_called_once_with(
        head_checkpoint_path="models/siglip_classifier_head.pt",
        siglip_model_id="google/siglip-so400m-patch14-384",
        minilm_model_id="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )


def test_load_model_siglip_classifier_backend_defaults():
    fake_model = mock.Mock()
    with mock.patch(
        "src.benchmark.models.siglip_classifier.SiglipClassifierModel.from_pretrained",
        return_value=fake_model,
    ) as mock_from_pretrained:
        load_model({"backend": "siglip_classifier"})

    mock_from_pretrained.assert_called_once_with(
        head_checkpoint_path="models/siglip_classifier_head.pt",
        siglip_model_id="google/siglip-so400m-patch14-384",
        minilm_model_id="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_run_benchmark_siglip_backend.py -v`
Expected: FAIL with `ValueError: Unknown backend: siglip_classifier`

- [ ] **Step 3: Add the backend branch**

In `src/benchmark/run_benchmark.py`, inside `load_model()`, add before the final `raise ValueError(...)` line (currently `run_benchmark.py:73`):

```python
    if backend == "siglip_classifier":
        from .models.siglip_classifier import SiglipClassifierModel
        return SiglipClassifierModel.from_pretrained(
            head_checkpoint_path=model_cfg.get(
                "head_checkpoint_path", "models/siglip_classifier_head.pt"
            ),
            siglip_model_id=model_cfg.get(
                "siglip_model_id", "google/siglip-so400m-patch14-384"
            ),
            minilm_model_id=model_cfg.get(
                "minilm_model_id", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            device=model_cfg.get("device", "cpu"),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_run_benchmark_siglip_backend.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/run_benchmark.py tests/test_run_benchmark_siglip_backend.py
git commit -m "feat: register siglip_classifier backend in run_benchmark load_model"
```

---

### Task 5: Stratified train/val/test split with persistence

**Files:**
- Create: `src/benchmark/train_classifier.py`
- Test: `tests/test_train_classifier.py`

**Interfaces:**
- Produces: `stratified_split(records: list[dict], seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, list[dict]]` returning `{"train": [...], "val": [...], "test": [...]}`, partitioning `records` with no overlap and preserving the `is_unanswerable` ratio in each split (within rounding).
- Produces: `save_split(split: dict[str, list[dict]], path: str) -> None` writes `{"train": [sample_id, ...], "val": [...], "test": [...]}` as JSON.
- Produces: `load_split(path: str, records: list[dict]) -> dict[str, list[dict]]` reads the JSON id lists and re-hydrates them back into full record dicts looked up from `records` (keyed by `sample_id`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_classifier.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.benchmark.train_classifier'`

- [ ] **Step 3: Implement the split functions**

Create `src/benchmark/train_classifier.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train_classifier.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/train_classifier.py tests/test_train_classifier.py
git commit -m "feat: add stratified train/val/test split for classifier training"
```

---

### Task 6: Embedding cache with content-hash invalidation

**Files:**
- Modify: `src/benchmark/train_classifier.py`
- Test: `tests/test_train_classifier.py`

**Interfaces:**
- Consumes: `ImageTextEncoder` protocol (Task 3) — any object with `encode_image(path) -> torch.Tensor` / `encode_text(text) -> torch.Tensor`; for mp_docvqa records, uses `encode_image_window(paths) -> torch.Tensor` if the encoder has it and the record's `metadata.window_pages` is non-empty.
- Produces: `compute_cache_key(records: list[dict], siglip_model_id: str, minilm_model_id: str) -> str` — a stable hash string derived from sorted sample ids + the two model ids (changes whenever the input set or encoder choice changes).
- Produces: `build_embedding_cache(records: list[dict], encoders, cache_path: str, cache_key: str) -> dict[str, dict]` — returns `{sample_id: {"image_embed": Tensor, "text_embed": Tensor, "label": bool}}`. If `cache_path` exists and its stored key matches `cache_key`, loads and returns it without calling the encoders. Otherwise builds it fresh (calling `encoders.encode_image`/`encode_text`/`encode_image_window`) and writes `{"cache_key": cache_key, "embeddings": {...}}` to `cache_path` via `torch.save`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train_classifier.py`:

```python
import torch

from src.benchmark.train_classifier import build_embedding_cache, compute_cache_key


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
        {"sample_id": "s1", "document_path": "p1.png", "question": "q1",
         "is_unanswerable": False, "metadata": {"window_pages": ["p1.png", "p2.png"]}},
    ]
    encoders = _FakeEncoders()
    cache_path = str(tmp_path / "cache.pt")
    key = compute_cache_key(records, "siglip-1", "minilm-1")

    cache = build_embedding_cache(records, encoders, cache_path, key)

    assert encoders.window_calls == [["p1.png", "p2.png"]]


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

    assert second_encoders.image_calls == ["doc1.png"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_classifier.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_embedding_cache'`

- [ ] **Step 3: Implement the cache functions**

Add to `src/benchmark/train_classifier.py` (top-level, alongside the split functions):

```python
import hashlib
from pathlib import Path

import torch


def compute_cache_key(records: list[dict], siglip_model_id: str, minilm_model_id: str) -> str:
    sample_ids = sorted(r["sample_id"] for r in records)
    payload = json.dumps([sample_ids, siglip_model_id, minilm_model_id], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_embedding_cache(
    records: list[dict],
    encoders,
    cache_path: str,
    cache_key: str,
) -> dict[str, dict]:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train_classifier.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/train_classifier.py tests/test_train_classifier.py
git commit -m "feat: add hash-gated embedding cache for classifier training"
```

---

### Task 7: Train/evaluate the head + CLI entry point + config

**Files:**
- Modify: `src/benchmark/train_classifier.py`
- Create: `configs/siglip_classifier_config.yaml`
- Test: `tests/test_train_classifier.py`

**Interfaces:**
- Consumes: `ClassifierHead` (Task 2), `stratified_split`/`load_split`/`save_split` (Task 5), `build_embedding_cache`/`compute_cache_key` (Task 6), `compute_metrics`/`BenchmarkMetrics` from `src/benchmark/evaluation/metrics.py`.
- Produces: `train_head(train_records: list[dict], val_records: list[dict], embeddings: dict[str, dict], epochs: int = 20, lr: float = 1e-3, batch_size: int = 64) -> ClassifierHead` — trains with Adam + `BCEWithLogitsLoss`, keeps the state dict with the best validation F1 (computed each epoch with `compute_metrics`), returns that best `ClassifierHead`.
- Produces: `evaluate_head(head: ClassifierHead, records: list[dict], embeddings: dict[str, dict]) -> "BenchmarkMetrics"` — runs the head over `records` in eval mode and returns metrics from `compute_metrics`.
- Produces: `main()` — CLI (`argparse`, single `--config` flag) that reads `configs/siglip_classifier_config.yaml`, runs the full pipeline (load data → split → cache → train → evaluate → save head + log to MLflow), invoked via `python -m src.benchmark.train_classifier --config configs/siglip_classifier_config.yaml`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_train_classifier.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_classifier.py -v`
Expected: FAIL with `ImportError: cannot import name 'train_head'`

- [ ] **Step 3: Implement `train_head`, `evaluate_head`, and `main`**

Add to `src/benchmark/train_classifier.py`:

```python
import argparse
import copy

import yaml

from .evaluation.metrics import BenchmarkMetrics, compute_metrics
from .models.siglip_classifier import ClassifierHead, PretrainedEncoders


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

    head_path = Path(config["head_checkpoint_path"])
    head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), head_path)
    print(f"Saved head to {head_path}")
    print(test_metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create the config file**

Create `configs/siglip_classifier_config.yaml`:

```yaml
dataset_paths:
  - data/final/docvqa_final.json
  - data/final/mp_docvqa_final.json

split_path: data/cache/siglip_classifier_split.json
cache_path: data/cache/siglip_embeddings.pt
head_checkpoint_path: models/siglip_classifier_head.pt

siglip_model_id: google/siglip-so400m-patch14-384
minilm_model_id: sentence-transformers/all-MiniLM-L6-v2
device: cpu

seed: 42
epochs: 20
lr: 0.001
batch_size: 64
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_train_classifier.py -v`
Expected: 11 passed

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/train_classifier.py configs/siglip_classifier_config.yaml tests/test_train_classifier.py
git commit -m "feat: add classifier head training, evaluation, and CLI entry point"
```

---

### Task 8: Manual end-to-end verification (real models, real data)

This task has no automated test — it exercises the real SigLIP/MiniLM downloads and the full dataset, which is too slow/network-dependent for the pytest suite.

- [ ] **Step 1: Run the full training pipeline**

```bash
uv run python -m src.benchmark.train_classifier --config configs/siglip_classifier_config.yaml
```

Expected: downloads `google/siglip-so400m-patch14-384` and `all-MiniLM-L6-v2` on first run, builds `data/cache/siglip_embeddings.pt`, trains for 20 epochs, prints a `BenchmarkMetrics` line, and writes `models/siglip_classifier_head.pt`.

- [ ] **Step 2: Add the backend to `configs/benchmark_config.yaml` and run a real benchmark pass**

Add to the `models:` list in `configs/benchmark_config.yaml`:

```yaml
  - backend: siglip_classifier
    head_checkpoint_path: models/siglip_classifier_head.pt
```

Run:
```bash
uv run python -m src.benchmark.run_benchmark --config configs/benchmark_config.yaml
```

Expected: a new `results/benchmark_docvqa/siglip_classifier_benchmark_result.json` appears with per-sample predictions and `inference_time_s` values, which should be visibly lower than the VLM backends' values in the same directory.

- [ ] **Step 3: Sanity-check inference speed and metrics**

Compare `inference_time_s` (mean) and `f1`/`mcc` from the new result file against an existing VLM result file in `results/benchmark_docvqa/`. Confirm the classifier is faster; note the F1 gap (if any) for the report.

- [ ] **Step 4: Commit the new config entry (not the model weights or cache — verify `.gitignore` covers them)**

```bash
git status
```

Confirm `models/siglip_classifier_head.pt` and `data/cache/*` are untracked-but-ignored (or add patterns to `.gitignore` if not already covered), then:

```bash
git add configs/benchmark_config.yaml
git commit -m "feat: enable siglip_classifier backend in benchmark config"
```
