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


import mlflow


def test_run_pipeline_creates_mlflow_run(tmp_path):
    """pipeline.run_pipeline() must create an MLflow run with expected params."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

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
            original_question="What year?",
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
    # all params present
    assert run.data.params["max_samples"] == "2"
    assert run.data.params["corruption_types"] == "nlp_entity"
    assert "use_judge" in run.data.params
    # top-level metrics present
    assert "total_samples" in run.data.metrics
    assert "nlp_entity_count" in run.data.metrics


def test_run_benchmark_creates_mlflow_run(tmp_path):
    """run_benchmark() must create one MLflow run per model with expected metrics."""
    import json
    import unittest.mock as mock
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlruns.db")

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
        sample_id="s1", predicted_unanswerable=True, confidence=-1.0, raw_response="UNANSWERABLE"
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
    assert run.data.params["backend"] == "vllm"
    assert "f1" in run.data.metrics
    assert "mcc" in run.data.metrics
    assert "specificity" in run.data.metrics
    assert "f1_nlp_entity" in run.data.metrics
