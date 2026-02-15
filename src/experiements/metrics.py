from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    per_class: Dict[str, Dict[str, float]]
    confusion: List[List[int]]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    per_class = {
        "0": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0]), "support": float(s[0])},
        "1": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f[1]), "support": float(s[1])},
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return Metrics(accuracy=acc, macro_f1=macro_f1, per_class=per_class, confusion=cm)
