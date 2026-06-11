"""Binary classification metrics for unanswerable question detection."""

import math
from collections import defaultdict
from dataclasses import dataclass
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
    y_true: Sequence[bool],   # True = actually unanswerable
    y_pred: Sequence[bool],   # True = predicted unanswerable
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
