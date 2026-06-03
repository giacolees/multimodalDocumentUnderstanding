"""Binary classification metrics for unanswerable question detection."""

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

    def __str__(self) -> str:
        return (
            f"Accuracy={self.accuracy:.3f}  Precision={self.precision:.3f}  "
            f"Recall={self.recall:.3f}  F1={self.f1:.3f}  "
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
    return BenchmarkMetrics(accuracy=accuracy, precision=precision,
                             recall=recall, f1=f1, tp=tp, fp=fp, tn=tn, fn=fn)
