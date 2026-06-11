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
