from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class Metrics:
    """Container for evaluation results."""
    accuracy: float
    macro_f1: float
    per_class: Dict[str, Dict[str, float]]
    confusion: List[List[int]]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Compute accuracy, macro-F1, per-class scores and confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    per_class = {
        "0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": float(support[0]),
        },
        "1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": float(support[1]),
        },
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return Metrics(
        accuracy=acc,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=cm,
    )
