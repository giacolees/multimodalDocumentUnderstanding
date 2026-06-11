# MLflow Experiment Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add inline MLflow experiment tracking to all three pipeline stages (corruption, benchmark, mitigation) so every run's params, metrics, and artifacts are recorded in a local `mlruns/` directory.

**Architecture:** Option A — no wrapper modules. `mlflow` calls go directly into `pipeline.py`, `run_benchmark.py`, and `run_mitigation.py`. Metrics are extended in `metrics.py` (new fields on `BenchmarkMetrics`, two new helpers). Three MLflow experiments: `dataset-corruption`, `benchmark`, `mitigation`.

**Tech Stack:** `mlflow`, `matplotlib` (already in deps), Python stdlib `math` and `datetime`.

---

## File Map

| File | Change |
|---|---|
| `pyproject.toml` | Add `mlflow` to `[project.dependencies]` |
| `.gitignore` | Add `mlruns/` |
| `src/benchmark/evaluation/metrics.py` | Add `specificity`, `balanced_accuracy`, `mcc` to `BenchmarkMetrics`; add `compute_per_type_metrics`; add `plot_confusion_matrix` |
| `src/dataset/pipeline.py` | Wrap `run_pipeline()` body with `mlflow.start_run()` |
| `src/benchmark/run_benchmark.py` | Wrap each model loop with `mlflow.start_run()` |
| `src/mitigation/run_mitigation.py` | Wrap each strategy loop with `mlflow.start_run()`; load baseline for delta metrics |
| `tests/test_mlflow_tracking.py` | Unit tests for new metrics + MLflow run assertions |

---

## Task 1: Add mlflow dependency and gitignore entry

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Add `mlflow` to pyproject.toml dependencies**

In `pyproject.toml`, add `"mlflow>=2.13"` to the `dependencies` list after `"matplotlib>=3.10.9"`:

```toml
dependencies = [
    "requests>=2.31.0",
    "pyyaml>=6.0",
    "pillow>=10.0",
    "datasets>=2.18.0",
    "pandas>=2.0",
    "pdf2image>=1.17",
    "pypdf>=4.0",
    "pydantic-ai>=1.0.0",
    "langchain-core>=0.3.0",
    "matplotlib>=3.10.9",
    "mlflow>=2.13",
]
```

- [ ] **Step 2: Add `mlruns/` to .gitignore**

Append to `.gitignore`:
```
mlruns/
```

- [ ] **Step 3: Sync deps**

```bash
uv sync
```

Expected: resolves without error, `mlflow` appears in `.venv`.

- [ ] **Step 4: Verify import**

```bash
uv run python -c "import mlflow; print(mlflow.__version__)"
```

Expected: prints a version string like `2.x.x`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore uv.lock
git commit -m "feat: add mlflow dependency"
```

---

## Task 2: Extend BenchmarkMetrics with new scalar metrics and helpers

**Files:**
- Modify: `src/benchmark/evaluation/metrics.py`
- Create: `tests/test_mlflow_tracking.py`

- [ ] **Step 1: Write failing tests for new metrics**

Create `tests/test_mlflow_tracking.py`:

```python
import math
import pytest
from src.benchmark.evaluation.metrics import (
    BenchmarkMetrics,
    compute_metrics,
    compute_per_type_metrics,
    plot_confusion_matrix,
)


def test_specificity():
    m = compute_metrics([True, True, False, False], [True, False, True, False])
    # TN=1, FP=1 → specificity = 0.5
    assert math.isclose(m.specificity, 0.5)


def test_balanced_accuracy():
    m = compute_metrics([True, True, False, False], [True, False, True, False])
    # recall=0.5, specificity=0.5 → balanced_accuracy=0.5
    assert math.isclose(m.balanced_accuracy, 0.5)


def test_mcc_perfect():
    m = compute_metrics([True, True, False, False], [True, True, False, False])
    assert math.isclose(m.mcc, 1.0)


def test_mcc_zero_denom():
    # all predictions positive → FN=0 FP=0 TN=0 — denom is 0
    m = compute_metrics([True, True], [True, True])
    assert m.mcc == 0.0


def test_compute_per_type_metrics():
    records = [
        {"corruption_type": "nlp_entity", "label_unanswerable": True,  "predicted_unanswerable": True},
        {"corruption_type": "nlp_entity", "label_unanswerable": True,  "predicted_unanswerable": False},
        {"corruption_type": "element",    "label_unanswerable": True,  "predicted_unanswerable": True},
    ]
    per_type = compute_per_type_metrics(records)
    assert "nlp_entity" in per_type
    assert "element" in per_type
    assert math.isclose(per_type["element"].f1, 1.0)
    assert math.isclose(per_type["nlp_entity"].recall, 0.5)


def test_plot_confusion_matrix_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    m = compute_metrics([True, True, False, False], [True, False, True, False])
    fig = plot_confusion_matrix(m, title="Test")
    assert isinstance(fig, Figure)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_mlflow_tracking.py -v
```

Expected: FAIL — `compute_per_type_metrics`, `plot_confusion_matrix` not defined; `BenchmarkMetrics` missing `specificity` etc.

- [ ] **Step 3: Implement extended metrics.py**

Replace `src/benchmark/evaluation/metrics.py` entirely with:

```python
"""Binary classification metrics for unanswerable question detection."""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class BenchmarkMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    specificity: float = 0.0
    balanced_accuracy: float = 0.0
    mcc: float = 0.0

    def __str__(self) -> str:
        return (
            f"Accuracy={self.accuracy:.3f}  Precision={self.precision:.3f}  "
            f"Recall={self.recall:.3f}  F1={self.f1:.3f}  "
            f"Specificity={self.specificity:.3f}  BalAcc={self.balanced_accuracy:.3f}  "
            f"MCC={self.mcc:.3f}  "
            f"(TP={self.tp} FP={self.fp} TN={self.tn} FN={self.fn})"
        )


def compute_metrics(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
) -> BenchmarkMetrics:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum(not t and p for t, p in zip(y_true, y_pred))
    tn = sum(not t and not p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))
    n = len(y_true)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return BenchmarkMetrics(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        tp=tp, fp=fp, tn=tn, fn=fn,
        specificity=specificity, balanced_accuracy=balanced_accuracy, mcc=mcc,
    )


def compute_per_type_metrics(records: list[dict]) -> dict[str, BenchmarkMetrics]:
    """Group records by corruption_type and return metrics for each group."""
    buckets: dict[str, tuple[list[bool], list[bool]]] = defaultdict(lambda: ([], []))
    for r in records:
        ctype = r.get("corruption_type", "unknown")
        buckets[ctype][0].append(r["label_unanswerable"])
        buckets[ctype][1].append(r["predicted_unanswerable"])
    return {ctype: compute_metrics(labels, preds) for ctype, (labels, preds) in buckets.items()}


def plot_confusion_matrix(m: BenchmarkMetrics, title: str):
    """Return a matplotlib Figure of the 2×2 confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(4, 4))
    cm = np.array([[m.tn, m.fp], [m.fn, m.tp]])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Answerable", "Unanswerable"])
    ax.set_yticklabels(["Answerable", "Unanswerable"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_mlflow_tracking.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/evaluation/metrics.py tests/test_mlflow_tracking.py
git commit -m "feat: extend BenchmarkMetrics with specificity, balanced_accuracy, mcc + helpers"
```

---

## Task 3: MLflow tracking in pipeline.py

**Files:**
- Modify: `src/dataset/pipeline.py`

- [ ] **Step 1: Write failing test for MLflow run creation**

Add to `tests/test_mlflow_tracking.py`:

```python
import tempfile
import mlflow


def test_run_pipeline_creates_mlflow_run(tmp_path):
    """pipeline.run_pipeline() must create an MLflow run with expected params."""
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))

    # Minimal config matching dataset_config.yaml structure
    config = {
        "corruption": {"max_samples": 2, "seed": 0, "distribution": {"nlp_entity": 1.0}},
        "loader": {},
        "quality_check": {},
    }

    # Patch loader and corruptors so no real data or model is needed
    import unittest.mock as mock
    from src.dataset.loaders.base_loader import QASample

    fake_sample = QASample(
        sample_id="s1",
        document_path="doc.png",
        question="What year?",
        answer="2020",
        page_index=0,
        metadata={},
    )

    with mock.patch("src.dataset.pipeline.LOADERS", {"fake": mock.MagicMock(return_value=mock.MagicMock(load=lambda: [fake_sample]))}), \
         mock.patch("src.dataset.pipeline.NLPEntityCorruptor") as MockNLP, \
         mock.patch("src.dataset.pipeline.LLMJudge"):
        from src.dataset.corruption.base_corruptor import CorruptedSample
        from src.dataset.corruption.base_corruptor import CorruptionType
        MockNLP.return_value.corrupt.return_value = CorruptedSample(
            corrupted_question="What year was it not?",
            corruption_type=CorruptionType.NLP_ENTITY,
            corruption_detail="year:2020→1999",
        )
        from src.dataset.pipeline import run_pipeline
        run_pipeline(
            dataset="fake",
            data_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            config=config,
            use_judge=False,
        )

    runs = mlflow.search_runs(experiment_names=["dataset-corruption"], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["dataset"] == "fake"
    assert "total_kept" in run.data.metrics
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_mlflow_tracking.py::test_run_pipeline_creates_mlflow_run -v
```

Expected: FAIL — `run_pipeline` does not call MLflow.

- [ ] **Step 3: Add MLflow tracking to pipeline.py**

Add these imports at the top of `src/dataset/pipeline.py` (after the existing imports):

```python
from datetime import datetime
import mlflow
```

Then wrap the body of `run_pipeline()` with an MLflow run. Replace the function signature and opening lines:

```python
def run_pipeline(
    dataset: str,
    data_dir: str,
    output_dir: str,
    config: dict,
    use_judge: bool = True,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    import inspect
    loader_cls = LOADERS[dataset]
    loader_kwargs = {
        k: v for k, v in config.get("loader", {}).items()
        if k in inspect.signature(loader_cls.__init__).parameters
    }
    loader = loader_cls(data_dir, **loader_kwargs)

    dist_cfg = config.get("corruption", {}).get("distribution", {})
    corruptor_map = {
        "nlp_entity": NLPEntityCorruptor,
        "element": ElementCorruptor,
        "layout": LayoutCorruptor,
    }
    dist_keys = [k for k in corruptor_map if k in dist_cfg]
    dist_weights = [dist_cfg[k] for k in dist_keys]
    corruptors = {k: corruptor_map[k](seed=seed) for k in dist_keys}
    qc = config.get("quality_check", {})
    judge = LLMJudge(
        model=qc.get("judge_model", "gemini-2.0-flash"),
        confidence_threshold=qc.get("confidence_threshold", 0.5),
        base_url=qc.get("judge_base_url") or None,
        max_retries=qc.get("max_retries", 3),
        max_tokens=qc.get("max_tokens", 2048),
    ) if use_judge else None

    log.info("Starting corruption pipeline: dataset=%s max_samples=%s judge=%s",
             dataset, config.get("corruption", {}).get("max_samples", "all"), "enabled" if judge else "disabled")

    all_samples = list(loader.load())
    max_samples = config.get("corruption", {}).get("max_samples", -1)
    if max_samples and max_samples > 0:
        all_samples = all_samples[:max_samples]
    log.info("Loaded %d samples", len(all_samples))

    output_path = Path(output_dir) / f"{dataset}_corrupted.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, sample in enumerate(all_samples):
        preferred = rng.choices(dist_keys, weights=dist_weights, k=1)[0]
        ordered = [preferred] + [k for k in dist_keys if k != preferred]
        shuffled = [corruptors[k] for k in ordered]
        state = {"sample": sample, "corruptors": shuffled, "judge": judge}
        record: Optional[dict] = _process_sample.invoke(state)
        if record is not None:
            results.append(record)
            log.info("[%s] ✓ kept [%s] %r → %r (total: %d)",
                     sample.sample_id,
                     record.get("corruption_type", "?"),
                     record.get("original_question"),
                     record.get("corrupted_question"),
                     len(results))
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        if (i + 1) % 50 == 0:
            log.info("Progress: %d/%d processed, %d kept", i + 1, len(all_samples), len(results))

    log.info("Done: %d/%d samples kept → %s", len(results), len(all_samples), output_path)

    # --- MLflow tracking ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow.set_experiment("dataset-corruption")
    with mlflow.start_run(run_name=f"{dataset}_{timestamp}"):
        mlflow.log_params({
            "dataset": dataset,
            "max_samples": config.get("corruption", {}).get("max_samples", -1),
            "window_size": config.get("loader", {}).get("window_size", 1),
            "corruption_types": ",".join(dist_keys),
            "use_judge": use_judge,
        })
        type_counts: dict[str, int] = {}
        for r in results:
            ct = r.get("corruption_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1
        mlflow.log_metrics({
            "total_samples": len(all_samples),
            "total_kept": len(results),
            **{f"{ct}_count": cnt for ct, cnt in type_counts.items()},
        })
        if output_path.exists():
            mlflow.log_artifact(str(output_path))

    return results
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_mlflow_tracking.py::test_run_pipeline_creates_mlflow_run -v
```

Expected: PASS.

- [ ] **Step 5: Run all tests**

```bash
uv run pytest tests/test_mlflow_tracking.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dataset/pipeline.py tests/test_mlflow_tracking.py
git commit -m "feat: add MLflow tracking to corruption pipeline"
```

---

## Task 4: MLflow tracking in run_benchmark.py

**Files:**
- Modify: `src/benchmark/run_benchmark.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_mlflow_tracking.py`:

```python
def test_run_benchmark_creates_mlflow_run(tmp_path):
    """run_benchmark() must create one MLflow run per model with expected metrics."""
    import json
    import unittest.mock as mock
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))

    dataset = [
        {
            "sample_id": "s1",
            "document_path": "doc.png",
            "question": "What year?",
            "is_unanswerable": True,
            "corruption_type": "nlp_entity",
        }
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    config = {
        "models": [{"backend": "vllm", "model_id": "test/model"}],
        "evaluation": {"metrics": ["accuracy", "f1"]},
    }

    from src.benchmark.models.base_model import PredictionResult
    mock_model = mock.MagicMock()
    mock_model.name.return_value = "test/model"
    mock_model.predict_unanswerable.return_value = PredictionResult(
        sample_id="s1", predicted_unanswerable=True, raw_response="UNANSWERABLE"
    )

    with mock.patch("src.benchmark.run_benchmark.load_model", return_value=mock_model):
        from src.benchmark.run_benchmark import run_benchmark
        run_benchmark(
            corrupted_dataset_path=str(dataset_path),
            config=config,
            output_dir=str(tmp_path / "results"),
        )

    runs = mlflow.search_runs(experiment_names=["benchmark"], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["model_id"] == "test/model"
    assert "f1" in run.data.metrics
    assert "mcc" in run.data.metrics
    assert "specificity" in run.data.metrics
    assert "f1_nlp_entity" in run.data.metrics
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_mlflow_tracking.py::test_run_benchmark_creates_mlflow_run -v
```

Expected: FAIL.

- [ ] **Step 3: Add imports to run_benchmark.py**

Add after the existing imports in `src/benchmark/run_benchmark.py`:

```python
from datetime import datetime
import mlflow
from .evaluation.metrics import compute_per_type_metrics, plot_confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

- [ ] **Step 4: Wrap the model loop with MLflow runs**

In `run_benchmark()`, add `mlflow.set_experiment("benchmark")` before the model loop, and wrap the per-model logic with `mlflow.start_run()`. The complete updated `run_benchmark` function:

```python
def run_benchmark(
    corrupted_dataset_path: str,
    config: dict,
    output_dir: str,
    prompt_template: str = _BASELINE_PROMPT,
):
    with open(corrupted_dataset_path) as f:
        dataset = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset_name = Path(corrupted_dataset_path).stem
    mlflow.set_experiment("benchmark")

    for model_cfg in config["models"]:
        model = load_model(model_cfg)
        model_name = model.name()
        safe_name = model_name.replace("/", "_")
        results_path = out / f"{safe_name}_benchmark_result.json"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with mlflow.start_run(run_name=f"{safe_name}_{dataset_name}_{timestamp}"):
            mlflow.log_params({
                "model_id": model_name,
                "backend": model_cfg["backend"],
                "dataset_path": corrupted_dataset_path,
                "num_samples": len(dataset),
            })

            existing: dict = {}
            if results_path.exists():
                with open(results_path) as f:
                    existing = json.load(f)

            existing_records: list[dict] = existing.get("records", [])
            done_ids = {r["sample_id"] for r in existing_records}
            remaining = [item for item in dataset if item["sample_id"] not in done_ids]

            if not remaining:
                print(f"\n[{model_name}] all {len(dataset)} samples already done, skipping")
                continue

            if done_ids:
                print(f"\n[{model_name}] resuming: {len(done_ids)} done, {len(remaining)} remaining")

            records = list(existing_records)
            for item in remaining:
                question = item["question"] if "question" in item else item["corrupted_question"]
                label: bool = item.get("is_unanswerable", True)
                result = model.predict_unanswerable(
                    document_path=item["document_path"],
                    question=question,
                    prompt_template=prompt_template,
                )
                result.sample_id = item["sample_id"]
                records.append({
                    "sample_id": item["sample_id"],
                    "predicted_unanswerable": result.predicted_unanswerable,
                    "label_unanswerable": label,
                    "raw_response": result.raw_response,
                    "corruption_type": item["corruption_type"],
                })
                preds = [r["predicted_unanswerable"] for r in records]
                labels = [r["label_unanswerable"] for r in records]
                metrics = compute_metrics(labels, preds)
                with open(results_path, "w") as f:
                    json.dump({"records": records, "metrics": metrics.__dict__}, f, indent=2)

            preds = [r["predicted_unanswerable"] for r in records]
            labels = [r["label_unanswerable"] for r in records]
            metrics = compute_metrics(labels, preds)
            print(f"\n[{model_name}] {metrics}")
            with open(results_path, "w") as f:
                json.dump({"records": records, "metrics": metrics.__dict__}, f, indent=2)
            print(f"Results saved → {results_path}")

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
            })

            per_type = compute_per_type_metrics(records)
            for ctype, tm in per_type.items():
                mlflow.log_metric(f"f1_{ctype}", tm.f1)

            fig = plot_confusion_matrix(metrics, title=model_name)
            mlflow.log_figure(fig, f"confusion_matrix_{safe_name}.png")
            plt.close(fig)

            mlflow.log_artifact(str(results_path))
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_mlflow_tracking.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/run_benchmark.py tests/test_mlflow_tracking.py
git commit -m "feat: add MLflow tracking to benchmark runner"
```

---

## Task 5: MLflow tracking in run_mitigation.py

**Files:**
- Modify: `src/mitigation/run_mitigation.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_mlflow_tracking.py`:

```python
def test_run_mitigation_creates_mlflow_run(tmp_path):
    """run_mitigation() must create one run per strategy with delta metrics."""
    import json
    import unittest.mock as mock
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))

    dataset = [
        {
            "sample_id": "s1",
            "document_path": "doc.png",
            "corrupted_question": "What year was it not?",
            "corruption_type": "nlp_entity",
        }
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    baseline = {"metrics": {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "mcc": 0.0}}
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))

    config = {
        "strategies": ["few_shot"],
        "model": {"backend": "vllm", "model_id": "test/model"},
    }

    from src.benchmark.models.base_model import PredictionResult
    mock_model = mock.MagicMock()
    mock_model.name.return_value = "test/model"
    mock_model.predict_unanswerable.return_value = PredictionResult(
        sample_id="s1", predicted_unanswerable=True, raw_response="UNANSWERABLE"
    )

    with mock.patch("src.mitigation.run_mitigation._load_model", return_value=mock_model):
        from src.mitigation.run_mitigation import run_mitigation
        run_mitigation(
            corrupted_dataset_path=str(dataset_path),
            baseline_results_path=str(baseline_path),
            config=config,
            output_dir=str(tmp_path / "results"),
            strategies=["few_shot"],
        )

    runs = mlflow.search_runs(experiment_names=["mitigation"], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["strategy"] == "few_shot"
    assert "f1" in run.data.metrics
    assert "delta_f1" in run.data.metrics
    assert "delta_mcc" in run.data.metrics
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_mlflow_tracking.py::test_run_mitigation_creates_mlflow_run -v
```

Expected: FAIL.

- [ ] **Step 3: Add imports to run_mitigation.py**

Add after existing imports in `src/mitigation/run_mitigation.py`:

```python
from datetime import datetime
import mlflow
from ..benchmark.evaluation.metrics import compute_per_type_metrics, plot_confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

- [ ] **Step 4: Wrap each strategy with an MLflow run**

Replace the `run_mitigation()` function with:

```python
def run_mitigation(
    corrupted_dataset_path: str,
    baseline_results_path: str,
    config: dict,
    output_dir: str,
    strategies: list[str] | None = None,
):
    with open(corrupted_dataset_path) as f:
        dataset: list[dict] = json.load(f)

    baseline_metrics: dict = {}
    if Path(baseline_results_path).exists():
        with open(baseline_results_path) as f:
            baseline_data = json.load(f)
        baseline_metrics = baseline_data.get("metrics", {})

    requested = strategies or config.get("strategies", list(_PROMPT_STRATEGIES.keys()))
    results = {}
    dataset_name = Path(corrupted_dataset_path).stem
    model_cfg = config.get("model", {})
    model_id = model_cfg.get("model_id", "unknown")
    mlflow.set_experiment("mitigation")

    # --- Prompt-based strategies ---
    prompt_strategies = [s for s in requested if s in _PROMPT_STRATEGIES]
    if prompt_strategies:
        model = _load_model(model_cfg)
        for strategy_name in prompt_strategies:
            prompt_fn = _PROMPT_STRATEGIES[strategy_name]
            preds, labels, records = [], [], []

            for item in dataset:
                prompt = prompt_fn(item["corrupted_question"], None)
                result = model.predict_unanswerable(
                    document_path=item["document_path"],
                    question=item["corrupted_question"],
                    prompt_template=prompt,
                )
                preds.append(result.predicted_unanswerable)
                labels.append(True)
                records.append({
                    "sample_id": item["sample_id"],
                    "strategy": strategy_name,
                    "predicted_unanswerable": result.predicted_unanswerable,
                    "label_unanswerable": True,
                    "raw_response": result.raw_response,
                    "corruption_type": item.get("corruption_type", "unknown"),
                })

            metrics = compute_metrics(labels, preds)
            print(f"\n[{strategy_name}] {metrics}")
            results[strategy_name] = {"records": records, "metrics": metrics.__dict__}

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with mlflow.start_run(run_name=f"{strategy_name}_{model_id}_{timestamp}"):
                mlflow.log_params({
                    "strategy": strategy_name,
                    "model_id": model_id,
                    "dataset_path": corrupted_dataset_path,
                    "num_samples": len(dataset),
                })
                delta_f1 = metrics.f1 - baseline_metrics.get("f1", 0.0)
                delta_mcc = metrics.mcc - baseline_metrics.get("mcc", 0.0)
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
                })
                per_type = compute_per_type_metrics(records)
                for ctype, tm in per_type.items():
                    mlflow.log_metric(f"f1_{ctype}", tm.f1)
                safe_model = model_id.replace("/", "_")
                fig = plot_confusion_matrix(metrics, title=f"{strategy_name} — {model_id}")
                mlflow.log_figure(fig, f"confusion_matrix_{strategy_name}_{safe_model}.png")
                plt.close(fig)

    # --- Fine-tuning strategy ---
    if "finetuning" in requested:
        ft_cfg_dict = config.get("finetuning", {})
        ft_config = FinetuningConfig(**ft_cfg_dict)
        ft_metrics = finetune(dataset, ft_config)
        results["finetuning"] = {"metrics": ft_metrics}

    # --- Persist ---
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "mitigation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out / 'mitigation_results.json'}")
```

- [ ] **Step 5: Run all tests**

```bash
uv run pytest tests/test_mlflow_tracking.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mitigation/run_mitigation.py tests/test_mlflow_tracking.py
git commit -m "feat: add MLflow tracking to mitigation runner"
```

---

## Task 6: Smoke-test the MLflow UI

- [ ] **Step 1: Start the MLflow UI**

```bash
uv run mlflow ui --port 5000
```

Expected: server starts, prints `Listening at: http://127.0.0.1:5000`.

- [ ] **Step 2: Verify the UI is reachable**

Open `http://127.0.0.1:5000` in a browser. You should see the MLflow Experiments page. Stop the server with `Ctrl+C`.

- [ ] **Step 3: Final commit**

```bash
git add docs/superpowers/plans/2026-06-11-mlflow-tracking.md
git commit -m "docs: add MLflow tracking implementation plan"
```
