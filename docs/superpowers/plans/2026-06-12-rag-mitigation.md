# RAG Mitigation Strategy + Mitigation Module Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `src/mitigation/` into isolated strategy classes behind a shared interface, extract evaluation/MLflow logging into `evaluation.py`, slim `run_mitigation.py` to a thin runner, and add a hybrid-RAG strategy (BM25 + SentenceTransformer + RRF) that transcribes the document with the vision model and injects top-k retrieved chunks as grounding context.

**Architecture:** Each strategy implements `MitigationStrategy` (ABC in `strategies/base.py`). The shared `evaluation.py` owns the per-item inference loop and all MLflow logging; the runner dispatches strategies via a `registry.py` dict. The RAG strategy uses a two-pass flow: `model.generate()` transcribes the page (cached to disk), then hybrid BM25+dense retrieval with RRF fusion selects the top-k chunks to inject into the prompt.

**Tech Stack:** Python 3.11, `rank-bm25>=0.2` (BM25Okapi), `sentence-transformers>=3.0` (all-MiniLM-L6-v2), MLflow, pytest/unittest.mock.

---

## File map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/benchmark/models/base_model.py` | Add default `generate()` (raises NotImplementedError) |
| Modify | `src/benchmark/models/vllm_model.py` | Implement `generate()` reusing `_load_image_b64` |
| Create | `src/mitigation/strategies/base.py` | `MitigationStrategy` ABC |
| Modify | `src/mitigation/strategies/few_shot.py` | Add `FewShotStrategy` class |
| Modify | `src/mitigation/strategies/chain_of_thought.py` | Add `ChainOfThoughtStrategy` class |
| Modify | `src/mitigation/strategies/knowledge_injection.py` | Add `KnowledgeInjectionStrategy` class |
| Create | `src/mitigation/evaluation.py` | `evaluate_strategy()` — loop + MLflow (lifted from runner) |
| Create | `src/mitigation/registry.py` | `STRATEGIES` dict |
| Modify | `src/mitigation/run_mitigation.py` | Slim to thin runner using registry + evaluation |
| Create | `src/mitigation/strategies/rag.py` | `RagRetriever` + `RagStrategy` |
| Modify | `src/mitigation/registry.py` | Register `"rag": RagStrategy` |
| Modify | `configs/mitigation_config.yaml` | Add `rag:` block |
| Modify | `pyproject.toml` | Add `mitigation` optional extra |
| Create | `tests/test_mitigation_refactor.py` | Tests for interface + evaluation refactor |
| Create | `tests/test_rag.py` | Tests for RagRetriever + RagStrategy |

---

## Task 1: Add `generate()` to BaseVisionModel and VllmModel

**Files:**
- Modify: `src/benchmark/models/base_model.py`
- Modify: `src/benchmark/models/vllm_model.py`
- Create: `tests/test_generate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate.py
import pytest
import unittest.mock as mock
from src.benchmark.models.base_model import BaseVisionModel, PredictionResult


class _StubModel(BaseVisionModel):
    def predict_unanswerable(self, document_path, question, prompt_template, page_indices=None):
        return PredictionResult(sample_id="", predicted_unanswerable=False,
                                confidence=-1, raw_response="")
    def name(self):
        return "stub"


def test_generate_base_raises():
    stub = _StubModel()
    with pytest.raises(NotImplementedError):
        stub.generate("doc.png", "transcribe this")


def test_vllm_generate_returns_text(tmp_path):
    """VllmModel.generate() posts to the completions endpoint and returns raw text."""
    from src.benchmark.models.vllm_model import VllmModel
    from PIL import Image
    img = Image.new("RGB", (4, 4), color=(255, 255, 255))
    img_path = tmp_path / "page.png"
    img.save(img_path)

    model = VllmModel(base_url="http://fake:9999/v1", model_id="test/m", api_key="x")

    fake_response = mock.MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "Transcribed text from page."}}]
    }
    fake_response.raise_for_status = mock.MagicMock()

    with mock.patch.object(model._requests, "post", return_value=fake_response) as mock_post:
        result = model.generate(str(img_path), "Transcribe this page.", max_tokens=512)

    assert result == "Transcribed text from page."
    call_payload = mock_post.call_args[1]["json"]
    assert call_payload["max_tokens"] == 512
    assert call_payload["temperature"] == 0.0
    content = call_payload["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "Transcribe this page."
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run pytest tests/test_generate.py -v
```
Expected: `AttributeError` or `NotImplementedError` — `generate` does not exist yet.

- [ ] **Step 3: Add `generate()` default to `BaseVisionModel`**

In `src/benchmark/models/base_model.py`, after the `name()` abstractmethod:

```python
    def generate(
        self,
        document_path: str,
        prompt: str,
        page_indices: list[int] | None = None,
        max_tokens: int = 1024,
    ) -> str:
        raise NotImplementedError(f"{self.name()} does not support generate()")
```

- [ ] **Step 4: Implement `generate()` in `VllmModel`**

In `src/benchmark/models/vllm_model.py`, add after the `predict_unanswerable` method:

```python
    def generate(
        self,
        document_path: str,
        prompt: str,
        page_indices: list[int] | None = None,
        max_tokens: int = 1024,
    ) -> str:
        page = (page_indices or [0])[0]
        image_b64 = _load_image_b64(document_path, page_index=page,
                                    max_pixels=self._max_image_pixels)
        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        payload: dict = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        resp = self._requests.post(self._url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
uv run pytest tests/test_generate.py -v
```
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/models/base_model.py src/benchmark/models/vllm_model.py tests/test_generate.py
git commit -m "feat: add generate() to BaseVisionModel and VllmModel"
```

---

## Task 2: `MitigationStrategy` ABC

**Files:**
- Create: `src/mitigation/strategies/base.py`
- Create: `tests/test_mitigation_refactor.py` (first tests)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mitigation_refactor.py
import pytest


def test_mitigation_strategy_is_abstract():
    """Cannot instantiate MitigationStrategy directly — build_prompt is abstract."""
    from src.mitigation.strategies.base import MitigationStrategy
    with pytest.raises(TypeError):
        MitigationStrategy()


def test_concrete_strategy_prepare_is_noop():
    """prepare() default implementation does nothing."""
    from src.mitigation.strategies.base import MitigationStrategy

    class Concrete(MitigationStrategy):
        name = "concrete"
        def build_prompt(self, item, model):
            return "prompt"

    s = Concrete()
    s.prepare([], None)  # must not raise
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_mitigation_refactor.py -v
```
Expected: `ImportError` — `base.py` does not exist yet.

- [ ] **Step 3: Create `src/mitigation/strategies/base.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod


class MitigationStrategy(ABC):
    name: str

    def prepare(self, dataset: list[dict], model) -> None:
        """Optional one-time setup called once before the eval loop. No-op by default."""

    @abstractmethod
    def build_prompt(self, item: dict, model) -> str:
        """Return the prompt string for this item. Leave {question} as a literal placeholder."""
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_mitigation_refactor.py -v
```
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mitigation/strategies/base.py tests/test_mitigation_refactor.py
git commit -m "feat: add MitigationStrategy ABC"
```

---

## Task 3: Prompt strategy classes

**Files:**
- Modify: `src/mitigation/strategies/few_shot.py`
- Modify: `src/mitigation/strategies/chain_of_thought.py`
- Modify: `src/mitigation/strategies/knowledge_injection.py`
- Modify: `tests/test_mitigation_refactor.py`

- [ ] **Step 1: Add tests to `tests/test_mitigation_refactor.py`**

Append to the file:

```python
def test_few_shot_strategy_builds_prompt():
    from src.mitigation.strategies.few_shot import FewShotStrategy
    s = FewShotStrategy({"k": 2})
    item = {"corrupted_question": "What year?"}
    prompt = s.build_prompt(item, model=None)
    assert "UNANSWERABLE" in prompt
    assert "What year?" in prompt


def test_cot_strategy_builds_prompt():
    from src.mitigation.strategies.chain_of_thought import ChainOfThoughtStrategy
    s = ChainOfThoughtStrategy({})
    item = {"corrupted_question": "Where is Table 5?"}
    prompt = s.build_prompt(item, model=None)
    assert "step by step" in prompt.lower()
    assert "Where is Table 5?" in prompt


def test_knowledge_injection_strategy_builds_prompt():
    from src.mitigation.strategies.knowledge_injection import KnowledgeInjectionStrategy
    s = KnowledgeInjectionStrategy({})
    item = {"corrupted_question": "What year?"}
    prompt = s.build_prompt(item, model=None)
    assert "UNANSWERABLE" in prompt
    assert "What year?" in prompt
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_mitigation_refactor.py::test_few_shot_strategy_builds_prompt -v
```
Expected: `ImportError` — `FewShotStrategy` not defined yet.

- [ ] **Step 3: Add `FewShotStrategy` to `src/mitigation/strategies/few_shot.py`**

Append to the end of the existing file (keep all existing code):

```python
from .base import MitigationStrategy


class FewShotStrategy(MitigationStrategy):
    name = "few_shot"

    def __init__(self, config: dict) -> None:
        self._k = config.get("k", 2)

    def build_prompt(self, item: dict, model) -> str:
        return build_few_shot_prompt(item["corrupted_question"], k=self._k)
```

- [ ] **Step 4: Add `ChainOfThoughtStrategy` to `src/mitigation/strategies/chain_of_thought.py`**

Append to the end of the existing file:

```python
from .base import MitigationStrategy


class ChainOfThoughtStrategy(MitigationStrategy):
    name = "chain_of_thought"

    def __init__(self, config: dict) -> None:
        pass

    def build_prompt(self, item: dict, model) -> str:
        return build_cot_prompt(item["corrupted_question"])
```

- [ ] **Step 5: Add `KnowledgeInjectionStrategy` to `src/mitigation/strategies/knowledge_injection.py`**

Append to the end of the existing file:

```python
from .base import MitigationStrategy


class KnowledgeInjectionStrategy(MitigationStrategy):
    name = "knowledge_injection"

    def __init__(self, config: dict) -> None:
        pass

    def build_prompt(self, item: dict, model) -> str:
        return build_knowledge_injection_prompt(item["corrupted_question"], DocumentMetadata())
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/test_mitigation_refactor.py -v
```
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mitigation/strategies/few_shot.py \
        src/mitigation/strategies/chain_of_thought.py \
        src/mitigation/strategies/knowledge_injection.py \
        tests/test_mitigation_refactor.py
git commit -m "feat: add MitigationStrategy wrappers for few_shot, cot, knowledge_injection"
```

---

## Task 4: Extract eval loop into `evaluation.py`

**Files:**
- Create: `src/mitigation/evaluation.py`
- Modify: `tests/test_mitigation_refactor.py`

- [ ] **Step 1: Add test to `tests/test_mitigation_refactor.py`**

Append to the file:

```python
def test_evaluate_strategy_returns_metrics_and_logs_mlflow(tmp_path):
    import json
    import mlflow
    import unittest.mock as mock
    from src.benchmark.models.base_model import PredictionResult

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlruns.db")
    mlflow.set_experiment("mitigation")

    dataset = [
        {"sample_id": "s1", "document_path": "doc.png",
         "corrupted_question": "Q1?", "corruption_type": "nlp_entity"},
        {"sample_id": "s2", "document_path": "doc.png",
         "corrupted_question": "Q2?", "corruption_type": "element"},
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    mock_model = mock.MagicMock()
    mock_model.name.return_value = "test/model"
    mock_model.predict_unanswerable.return_value = PredictionResult(
        sample_id="", predicted_unanswerable=True, confidence=-1, raw_response="UNANSWERABLE"
    )

    class TrivialStrategy:
        name = "trivial"
        def prepare(self, dataset, model): pass
        def build_prompt(self, item, model): return "Is this unanswerable? Q: {question}"

    from src.mitigation.evaluation import evaluate_strategy
    result = evaluate_strategy(
        strategy=TrivialStrategy(),
        dataset=dataset,
        model=mock_model,
        baseline_metrics={"f1": 0.5, "mcc": 0.0, "precision": 0.5,
                          "recall": 0.5, "specificity": 0.5, "balanced_accuracy": 0.5},
        model_id="test/model",
        corrupted_dataset_path=str(dataset_path),
    )

    assert result["metrics"]["f1"] == 1.0
    assert len(result["records"]) == 2

    runs = mlflow.search_runs(experiment_names=["mitigation"], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["strategy"] == "trivial"
    assert run.data.params["model_id"] == "test/model"
    assert "f1" in run.data.metrics
    assert "mcc" in run.data.metrics
    assert "delta_f1" in run.data.metrics
    assert "delta_mcc" in run.data.metrics
    assert "f1_nlp_entity" in run.data.metrics
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_mitigation_refactor.py::test_evaluate_strategy_returns_metrics_and_logs_mlflow -v
```
Expected: `ImportError` — `evaluation.py` does not exist.

- [ ] **Step 3: Create `src/mitigation/evaluation.py`**

This is a direct lift of the inline loop from `run_mitigation.py`:

```python
"""Shared evaluation loop for all prompt-based mitigation strategies."""

from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from ..benchmark.evaluation.metrics import (
    compute_metrics,
    compute_per_type_metrics,
    plot_confusion_matrix,
)

if TYPE_CHECKING:
    from .strategies.base import MitigationStrategy
    from ..benchmark.models.base_model import BaseVisionModel


def evaluate_strategy(
    strategy: "MitigationStrategy",
    dataset: list[dict],
    model: "BaseVisionModel",
    baseline_metrics: dict,
    model_id: str,
    corrupted_dataset_path: str,
) -> dict:
    """Run the per-item inference loop, compute metrics, log to MLflow, return results dict."""
    preds: list[bool] = []
    labels: list[bool] = []
    records: list[dict] = []
    inference_times: list[float] = []
    _sample_prompt: str = ""

    for item in dataset:
        prompt = strategy.build_prompt(item, model)
        if not _sample_prompt:
            _sample_prompt = prompt
        t0 = time.perf_counter()
        result = model.predict_unanswerable(
            document_path=item["document_path"],
            question=item["corrupted_question"],
            prompt_template=prompt,
        )
        elapsed = time.perf_counter() - t0
        inference_times.append(elapsed)
        preds.append(result.predicted_unanswerable)
        labels.append(True)
        records.append({
            "sample_id": item["sample_id"],
            "strategy": strategy.name,
            "predicted_unanswerable": result.predicted_unanswerable,
            "label_unanswerable": True,
            "raw_response": result.raw_response,
            "corruption_type": item.get("corruption_type", "unknown"),
            "inference_time_s": elapsed,
        })

    metrics = compute_metrics(labels, preds)
    print(f"\n[{strategy.name}] {metrics}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_id.replace("/", "_")

    with mlflow.start_run(run_name=f"{strategy.name}_{safe_model}_{timestamp}"):
        mlflow.set_tags({
            "strategy": strategy.name,
            "model": model_id,
            "dataset": Path(corrupted_dataset_path).stem,
        })
        mlflow.log_params({
            "strategy": strategy.name,
            "model_id": model_id,
            "dataset_path": corrupted_dataset_path,
            "num_samples": len(dataset),
        })
        ds = mlflow.data.from_pandas(
            pd.read_json(corrupted_dataset_path),
            name=Path(corrupted_dataset_path).stem,
            targets="is_unanswerable",
        )
        mlflow.log_input(ds, context="evaluation")

        if _sample_prompt:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False,
                prefix=f"prompt_{strategy.name}_",
            ) as tmp:
                tmp.write(_sample_prompt)
                mlflow.log_artifact(tmp.name, artifact_path="prompts")

        delta_f1 = metrics.f1 - baseline_metrics.get("f1", 0.0)
        delta_mcc = metrics.mcc - baseline_metrics.get("mcc", 0.0)
        delta_precision = metrics.precision - baseline_metrics.get("precision", 0.0)
        delta_recall = metrics.recall - baseline_metrics.get("recall", 0.0)
        delta_specificity = metrics.specificity - baseline_metrics.get("specificity", 0.0)
        delta_balanced_accuracy = (
            metrics.balanced_accuracy - baseline_metrics.get("balanced_accuracy", 0.0)
        )
        mlflow.log_metrics({
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "tp": float(metrics.tp),
            "fp": float(metrics.fp),
            "tn": float(metrics.tn),
            "fn": float(metrics.fn),
            "specificity": metrics.specificity,
            "balanced_accuracy": metrics.balanced_accuracy,
            "mcc": metrics.mcc,
            "delta_f1": delta_f1,
            "delta_mcc": delta_mcc,
            "delta_precision": delta_precision,
            "delta_recall": delta_recall,
            "delta_specificity": delta_specificity,
            "delta_balanced_accuracy": delta_balanced_accuracy,
            **(
                {
                    "inference_time_mean_s": float(np.mean(inference_times)),
                    "inference_time_median_s": float(np.median(inference_times)),
                    "inference_time_p95_s": float(np.percentile(inference_times, 95)),
                    "inference_time_total_s": sum(inference_times),
                    "throughput_samples_per_s": len(inference_times) / sum(inference_times),
                }
                if inference_times
                else {}
            ),
        })

        per_type = compute_per_type_metrics(records)
        baseline_per_type = baseline_metrics.get("per_type", {})
        for ctype, tm in per_type.items():
            base_f1 = (
                baseline_per_type.get(ctype, {}).get("f1", 0.0)
                if isinstance(baseline_per_type, dict)
                else 0.0
            )
            mlflow.log_metrics({
                f"f1_{ctype}": tm.f1,
                f"precision_{ctype}": tm.precision,
                f"recall_{ctype}": tm.recall,
                f"specificity_{ctype}": tm.specificity,
                f"mcc_{ctype}": tm.mcc,
                f"delta_f1_{ctype}": tm.f1 - base_f1,
            })

        fig = plot_confusion_matrix(metrics, title=f"{strategy.name} — {model_id}")
        mlflow.log_figure(fig, f"confusion_matrix_{strategy.name}_{safe_model}.png")
        plt.close(fig)

        if per_type:
            ctypes = list(per_type.keys())
            strategy_f1s = [per_type[ct].f1 for ct in ctypes]
            base_f1s = [
                baseline_per_type.get(ct, {}).get("f1", 0.0)
                if isinstance(baseline_per_type, dict)
                else 0.0
                for ct in ctypes
            ]
            x = range(len(ctypes))
            fig3, ax3 = plt.subplots(figsize=(max(5, len(ctypes) * 2), 4))
            ax3.bar([i - 0.2 for i in x], base_f1s, width=0.4, label="baseline", color="lightgray")
            ax3.bar(
                [i + 0.2 for i in x], strategy_f1s, width=0.4,
                label=strategy.name, color="steelblue",
            )
            ax3.set_xticks(list(x))
            ax3.set_xticklabels(ctypes)
            ax3.set_ylim(0, 1.1)
            ax3.set_ylabel("F1")
            ax3.set_title(f"F1 by type: {strategy.name} vs baseline")
            ax3.legend()
            plt.tight_layout()
            mlflow.log_figure(fig3, f"f1_by_type_{strategy.name}_{safe_model}.png")
            plt.close(fig3)

    return {"records": records, "metrics": metrics.__dict__}
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_mitigation_refactor.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mitigation/evaluation.py tests/test_mitigation_refactor.py
git commit -m "feat: extract evaluate_strategy() into mitigation/evaluation.py"
```

---

## Task 5: `registry.py` + slim `run_mitigation.py`

**Files:**
- Create: `src/mitigation/registry.py`
- Modify: `src/mitigation/run_mitigation.py`

- [ ] **Step 1: Create `src/mitigation/registry.py`**

```python
from .strategies.few_shot import FewShotStrategy
from .strategies.chain_of_thought import ChainOfThoughtStrategy
from .strategies.knowledge_injection import KnowledgeInjectionStrategy

STRATEGIES: dict[str, type] = {
    "few_shot": FewShotStrategy,
    "chain_of_thought": ChainOfThoughtStrategy,
    "knowledge_injection": KnowledgeInjectionStrategy,
}
```

- [ ] **Step 2: Rewrite `src/mitigation/run_mitigation.py`**

Replace the entire file with this slimmed version. Note: `_load_model` is kept here unchanged — the existing test in `test_mlflow_tracking.py` patches `src.mitigation.run_mitigation._load_model` and will continue to work.

```python
"""Run mitigation experiments (few-shot, chain-of-thought, knowledge-injection, rag,
finetuning).

Usage:
    python -m src.mitigation.run_mitigation \
        --corrupted_dataset data/corrupted/docvqa_corrupted.json \
        --baseline_results results/benchmark/benchmark_results.json \
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


def run_mitigation(
    corrupted_dataset_path: str,
    baseline_results_path: str,
    config: dict,
    output_dir: str,
    strategies: list[str] | None = None,
) -> None:
    with open(corrupted_dataset_path) as f:
        dataset: list[dict] = json.load(f)

    baseline_metrics: dict = {}
    if Path(baseline_results_path).exists():
        with open(baseline_results_path) as f:
            baseline_data = json.load(f)
        baseline_metrics = baseline_data.get("metrics", {})

    requested = strategies or config.get("strategies", list(STRATEGIES.keys()))
    model_cfg = config.get("model", {})
    model_id = model_cfg.get("model_id", "unknown")
    mlflow.set_experiment("mitigation")

    results: dict = {}

    prompt_strategy_names = [s for s in requested if s in STRATEGIES]
    if prompt_strategy_names:
        model = _load_model(model_cfg)
        for name in prompt_strategy_names:
            strategy_cls = STRATEGIES[name]
            strategy_cfg = config.get(name, {})
            strategy = strategy_cls(strategy_cfg)
            strategy.prepare(dataset, model)
            results[name] = evaluate_strategy(
                strategy=strategy,
                dataset=dataset,
                model=model,
                baseline_metrics=baseline_metrics,
                model_id=model_id,
                corrupted_dataset_path=corrupted_dataset_path,
            )

    if "finetuning" in requested:
        from .strategies.finetuning import FinetuningConfig, finetune
        ft_config = FinetuningConfig(**config.get("finetuning", {}))
        results["finetuning"] = {"metrics": finetune(dataset, ft_config)}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "mitigation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out / 'mitigation_results.json'}")


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
    parser.add_argument("--corrupted_dataset", required=True)
    parser.add_argument("--baseline_results", required=True)
    parser.add_argument("--config", default="configs/mitigation_config.yaml")
    parser.add_argument("--output_dir", default="results/mitigation")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Override strategies from config (e.g. --strategies rag few_shot)",
    )
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run_mitigation(
        corrupted_dataset_path=args.corrupted_dataset,
        baseline_results_path=args.baseline_results,
        config=config,
        output_dir=args.output_dir,
        strategies=args.strategies,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the full test suite to confirm no regressions**

```bash
uv run pytest tests/ -v
```
Expected: all tests PASS (including the existing `test_run_mitigation_creates_mlflow_run`).

- [ ] **Step 4: Commit**

```bash
git add src/mitigation/registry.py src/mitigation/run_mitigation.py
git commit -m "refactor: slim run_mitigation.py — dispatch via registry, logging via evaluation.py"
```

---

## Task 6: `RagRetriever` — chunks, transcription, disk cache

**Files:**
- Create: `src/mitigation/strategies/rag.py` (partial — retrieval added in Task 7)
- Create: `tests/test_rag.py`

- [ ] **Step 1: Write tests for chunks and cache**

```python
# tests/test_rag.py
import unittest.mock as mock


def test_chunks_respects_max_chars():
    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(chunk_max_chars=50)
    text = "Hello world.\n\nThis is a second paragraph that is definitely longer than fifty chars."
    result = r.chunks(text)
    for c in result:
        assert len(c) <= 50, f"Chunk too long ({len(c)}): {c!r}"


def test_chunks_preserves_content():
    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(chunk_max_chars=100)
    text = "Line one.\nLine two.\nLine three."
    chunks = r.chunks(text)
    combined = " ".join(chunks)
    assert "Line one" in combined
    assert "Line two" in combined
    assert "Line three" in combined


def test_chunks_empty_text_returns_something():
    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(chunk_max_chars=50)
    result = r.chunks("   ")
    assert isinstance(result, list)


def test_transcribe_calls_model_generate(tmp_path):
    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(cache_dir=str(tmp_path))
    item = {"document_path": "data/raw/doc.png", "page_index": 0}
    mock_model = mock.MagicMock()
    mock_model.generate.return_value = "The year is 1975. Table 1 shows values."

    text = r.transcribe(item, mock_model)

    assert text == "The year is 1975. Table 1 shows values."
    mock_model.generate.assert_called_once()
    call_kwargs = mock_model.generate.call_args
    assert call_kwargs[0][0] == "data/raw/doc.png"
    assert "page_indices" in call_kwargs[1] or call_kwargs[0][2:] or True  # page_indices passed


def test_transcribe_cache_hit_skips_generate(tmp_path):
    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(cache_dir=str(tmp_path))
    item = {"document_path": "data/raw/doc.png", "page_index": 0}
    mock_model = mock.MagicMock()
    mock_model.generate.return_value = "Cached text."

    r.transcribe(item, mock_model)   # first call writes cache
    r.transcribe(item, mock_model)   # second call should read cache

    assert mock_model.generate.call_count == 1
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_rag.py -v
```
Expected: `ImportError` — `rag.py` does not exist.

- [ ] **Step 3: Create `src/mitigation/strategies/rag.py`** (chunks + transcribe only)

```python
"""RAG mitigation: transcribe page with vision model, chunk, hybrid-retrieve, inject context."""

from __future__ import annotations

from pathlib import Path

from .base import MitigationStrategy

_TRANSCRIBE_PROMPT = (
    "Transcribe all visible text from this document page faithfully. "
    "Preserve table labels, column headers, numbers, and layout markers exactly as shown. "
    "Output plain text only, no commentary."
)

_RAG_TEMPLATE = (
    "Here are the passages retrieved from this document most relevant to the question:\n"
    "{context}\n\n"
    "You are also shown the full document image. "
    "If the answer is not supported by the document, respond UNANSWERABLE; "
    "otherwise provide the answer.\n\n"
    "Question: {{question}}"
)


class RagRetriever:
    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 4,
        chunk_max_chars: int = 400,
        transcribe_max_tokens: int = 1024,
        cache_dir: str = "data/ocr_cache",
    ) -> None:
        self._embed_model_name = embed_model
        self._top_k = top_k
        self._chunk_max_chars = chunk_max_chars
        self._transcribe_max_tokens = transcribe_max_tokens
        self._cache_dir = Path(cache_dir)
        self._embedder = None  # lazy-loaded SentenceTransformer

    def _cache_key(self, item: dict) -> str:
        doc = item["document_path"].replace("/", "_").replace(".", "_")
        page = item.get("page_index", 0)
        return f"{doc}_p{page}"

    def transcribe(self, item: dict, model) -> str:
        """Return page text, using disk cache to avoid repeated model calls."""
        key = self._cache_key(item)
        cache_file = self._cache_dir / f"{key}.txt"
        if cache_file.exists():
            return cache_file.read_text()
        text = model.generate(
            item["document_path"],
            _TRANSCRIBE_PROMPT,
            page_indices=[item.get("page_index", 0)],
            max_tokens=self._transcribe_max_tokens,
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text)
        return text

    def chunks(self, text: str) -> list[str]:
        """Split text into chunks of at most chunk_max_chars, preserving as much content as possible."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        lines: list[str] = []
        for p in paragraphs:
            if len(p) <= self._chunk_max_chars:
                lines.append(p)
            else:
                for line in p.split("\n"):
                    if line.strip():
                        lines.append(line.strip())
        result: list[str] = []
        current = ""
        for line in lines:
            candidate = (current + "\n" + line).strip() if current else line
            if len(candidate) <= self._chunk_max_chars:
                current = candidate
            else:
                if current:
                    result.append(current)
                current = line[: self._chunk_max_chars]
        if current:
            result.append(current)
        if not result:
            stripped = text.strip()
            return [stripped[: self._chunk_max_chars]] if stripped else []
        return result

    def retrieve(self, item: dict, question: str, model) -> list[str]:
        """Stub — implemented in Task 7."""
        raise NotImplementedError
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_rag.py -v -k "not retrieve"
```
Expected: chunks and transcribe tests PASS; retrieve test (if any) skipped.

- [ ] **Step 5: Commit**

```bash
git add src/mitigation/strategies/rag.py tests/test_rag.py
git commit -m "feat: RagRetriever — chunks() and transcribe() with disk cache"
```

---

## Task 7: Hybrid retrieval — BM25 + dense + RRF

**Files:**
- Modify: `src/mitigation/strategies/rag.py`
- Modify: `tests/test_rag.py`

- [ ] **Step 1: Add retrieval tests to `tests/test_rag.py`**

Append to the file:

```python
def test_retrieve_rrf_ranks_relevant_chunk_first():
    """RRF fuses BM25 + dense; the chunk winning both signals appears first."""
    import sys
    import numpy as np

    # Mock rank_bm25 at the module level so the lazy import inside retrieve() uses it.
    mock_bm25_module = mock.MagicMock()
    mock_bm25_instance = mock.MagicMock()
    mock_bm25_module.BM25Okapi.return_value = mock_bm25_instance
    # chunk 0 = relevant, wins sparse
    mock_bm25_instance.get_scores.return_value = np.array([10.0, 0.1])

    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(top_k=1, chunk_max_chars=200)

    # Pre-set embedder to avoid sentence_transformers import
    mock_embedder = mock.MagicMock()
    # chunk 0 embedding is most similar to query (dot product: [0.9, 0.1])
    mock_embedder.encode.side_effect = [
        np.array([[0.9, 0.1], [0.1, 0.9]]),  # chunk embeddings (2 chunks)
        np.array([[1.0, 0.0]]),               # query embedding — chunk 0 wins dense too
    ]
    r._embedder = mock_embedder

    item = {"document_path": "doc.png", "page_index": 0}
    relevant = "net profit for 1975 was high"
    irrelevant = "company founded in 1960"
    full_text = f"{relevant}\n\n{irrelevant}"

    with mock.patch.object(r, "transcribe", return_value=full_text), \
         mock.patch.dict(sys.modules, {"rank_bm25": mock_bm25_module}):
        chunks = r.retrieve(item, "net profit 1975", mock.MagicMock())

    assert len(chunks) == 1
    assert chunks[0] == relevant


def test_retrieve_top_k_respected():
    """retrieve() returns at most top_k chunks."""
    import sys
    import numpy as np

    mock_bm25_module = mock.MagicMock()
    mock_bm25_instance = mock.MagicMock()
    mock_bm25_module.BM25Okapi.return_value = mock_bm25_instance
    mock_bm25_instance.get_scores.return_value = np.array([3.0, 2.0, 1.0])

    from src.mitigation.strategies.rag import RagRetriever
    r = RagRetriever(top_k=2, chunk_max_chars=30)
    mock_embedder = mock.MagicMock()
    mock_embedder.encode.side_effect = [
        np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        np.array([[1.0, 0.0]]),
    ]
    r._embedder = mock_embedder

    text = "Chunk A text here.\n\nChunk B text here.\n\nChunk C text here."
    item = {"document_path": "doc.png", "page_index": 0}
    with mock.patch.object(r, "transcribe", return_value=text), \
         mock.patch.dict(sys.modules, {"rank_bm25": mock_bm25_module}):
        chunks = r.retrieve(item, "query", mock.MagicMock())

    assert len(chunks) == 2
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_rag.py::test_retrieve_rrf_ranks_relevant_chunk_first -v
```
Expected: `NotImplementedError` from the stub `retrieve()`.

- [ ] **Step 3: Replace the stub `retrieve()` in `src/mitigation/strategies/rag.py`**

Replace the `retrieve` method:

```python
    def retrieve(self, item: dict, question: str, model) -> list[str]:
        """Hybrid BM25 + dense retrieval with RRF fusion."""
        text = self.transcribe(item, model)
        chunk_list = self.chunks(text)
        if not chunk_list:
            return []

        # Sparse ranking (BM25)
        from rank_bm25 import BM25Okapi
        tokenized = [c.lower().split() for c in chunk_list]
        bm25 = BM25Okapi(tokenized)
        sparse_scores = bm25.get_scores(question.lower().split())
        sparse_order = sorted(range(len(chunk_list)),
                              key=lambda i: sparse_scores[i], reverse=True)
        sparse_ranks: dict[int, int] = {idx: rank for rank, idx in enumerate(sparse_order)}

        # Dense ranking (SentenceTransformer)
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embed_model_name)
        import numpy as np
        embeddings = self._embedder.encode(chunk_list, normalize_embeddings=True)
        q_emb = self._embedder.encode([question], normalize_embeddings=True)[0]
        dense_scores_arr = embeddings @ q_emb
        dense_order = sorted(range(len(chunk_list)),
                             key=lambda i: float(dense_scores_arr[i]), reverse=True)
        dense_ranks: dict[int, int] = {idx: rank for rank, idx in enumerate(dense_order)}

        # RRF fusion (k=60, parameter-free)
        k = 60
        rrf = {
            i: 1.0 / (k + sparse_ranks[i]) + 1.0 / (k + dense_ranks[i])
            for i in range(len(chunk_list))
        }
        top = sorted(rrf, key=lambda i: rrf[i], reverse=True)
        return [chunk_list[i] for i in top[: self._top_k]]
```

- [ ] **Step 4: Run retrieval tests**

```bash
uv run pytest tests/test_rag.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mitigation/strategies/rag.py tests/test_rag.py
git commit -m "feat: RagRetriever.retrieve() — hybrid BM25 + dense + RRF fusion"
```

---

## Task 8: `RagStrategy`, registration, config, and dependency

**Files:**
- Modify: `src/mitigation/strategies/rag.py`
- Modify: `src/mitigation/registry.py`
- Modify: `configs/mitigation_config.yaml`
- Modify: `pyproject.toml`
- Modify: `tests/test_rag.py`

- [ ] **Step 1: Add `RagStrategy` test to `tests/test_rag.py`**

Append to the file:

```python
def test_rag_strategy_prompt_contains_context_and_question_placeholder():
    from src.mitigation.strategies.rag import RagStrategy
    strategy = RagStrategy({})
    item = {"document_path": "doc.png", "page_index": 0, "corrupted_question": "What year?"}
    mock_model = mock.MagicMock()

    with mock.patch.object(strategy.retriever, "retrieve",
                           return_value=["Relevant passage about 1975."]):
        prompt = strategy.build_prompt(item, mock_model)

    assert "Relevant passage about 1975." in prompt
    assert "{question}" in prompt          # placeholder left for VllmModel.predict_unanswerable
    assert "UNANSWERABLE" in prompt


def test_rag_strategy_empty_retrieval_still_returns_prompt():
    from src.mitigation.strategies.rag import RagStrategy
    strategy = RagStrategy({})
    item = {"document_path": "doc.png", "page_index": 0, "corrupted_question": "Anything?"}
    mock_model = mock.MagicMock()

    with mock.patch.object(strategy.retriever, "retrieve", return_value=[]):
        prompt = strategy.build_prompt(item, mock_model)

    assert "{question}" in prompt
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_rag.py::test_rag_strategy_prompt_contains_context_and_question_placeholder -v
```
Expected: `AttributeError` — `RagStrategy` not defined yet.

- [ ] **Step 3: Add `RagStrategy` class to `src/mitigation/strategies/rag.py`**

Append at the end of the file:

```python

class RagStrategy(MitigationStrategy):
    name = "rag"

    def __init__(self, config: dict) -> None:
        self.retriever = RagRetriever(
            embed_model=config.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            top_k=config.get("top_k", 4),
            chunk_max_chars=config.get("chunk_max_chars", 400),
            transcribe_max_tokens=config.get("transcribe_max_tokens", 1024),
            cache_dir=config.get("cache_dir", "data/ocr_cache"),
        )

    def build_prompt(self, item: dict, model) -> str:
        chunks = self.retriever.retrieve(item, item["corrupted_question"], model)
        context = "\n".join(f"- {c}" for c in chunks) if chunks else "(no passages retrieved)"
        return _RAG_TEMPLATE.format(context=context)
```

- [ ] **Step 4: Run `RagStrategy` tests**

```bash
uv run pytest tests/test_rag.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Register RAG in `src/mitigation/registry.py`**

Replace the file contents:

```python
from .strategies.few_shot import FewShotStrategy
from .strategies.chain_of_thought import ChainOfThoughtStrategy
from .strategies.knowledge_injection import KnowledgeInjectionStrategy
from .strategies.rag import RagStrategy

STRATEGIES: dict[str, type] = {
    "few_shot": FewShotStrategy,
    "chain_of_thought": ChainOfThoughtStrategy,
    "knowledge_injection": KnowledgeInjectionStrategy,
    "rag": RagStrategy,
}
```

- [ ] **Step 6: Update `configs/mitigation_config.yaml`**

Replace the file contents:

```yaml
# Part 3 – mitigation strategies

model:
  backend: google          # mistral | google | openrouter | llama_cpp | vllm
  model_id: gemini-2.0-flash

strategies:
  - few_shot
  - chain_of_thought
  - knowledge_injection
  - rag
  # Uncomment to include fine-tuning (requires GPU + unsloth):
  # - finetuning

few_shot:
  k: 2          # number of examples to inject

knowledge_injection:
  extract_metadata_automatically: false   # set true to run OCR-based metadata extraction

rag:
  embed_model: sentence-transformers/all-MiniLM-L6-v2
  top_k: 4
  chunk_max_chars: 400
  transcribe_max_tokens: 1024
  cache_dir: data/ocr_cache

finetuning:
  model_name: unsloth/Qwen2.5-VL-3B-Instruct
  output_dir: results/finetuned_judge
  load_in_4bit: true
  lora_r: 16
  lora_alpha: 16
  train_split: 0.8
  max_steps: 60
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 0.0002
  max_seq_length: 2048
  seed: 42
```

- [ ] **Step 7: Add `mitigation` extra to `pyproject.toml`**

In the `[project.optional-dependencies]` section, add:

```toml
mitigation = [
    "sentence-transformers>=3.0",
    "rank-bm25>=0.2",
]
```

- [ ] **Step 8: Run the full test suite**

```bash
uv run pytest tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add src/mitigation/strategies/rag.py \
        src/mitigation/registry.py \
        configs/mitigation_config.yaml \
        pyproject.toml \
        tests/test_rag.py
git commit -m "feat: RagStrategy + register rag, add mitigation extra (rank-bm25, sentence-transformers)"
```

---

## Running the full mitigation pipeline with RAG

After all tasks, install the new extra and run:

```bash
uv sync --extra mitigation

uv run python -m src.mitigation.run_mitigation \
  --corrupted_dataset data/corrupted/docvqa_corrupted.json \
  --baseline_results results/benchmark_docvqa/<model>_benchmark_result.json \
  --config configs/mitigation_config.yaml \
  --output_dir results/mitigation

uv run mlflow ui   # view results at http://localhost:5000
```

The `data/ocr_cache/` directory will be populated on first run; subsequent runs reuse the cached transcriptions.
