from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_preds_csv(path: str | Path, ids: List[int], y_true: List[int], y_pred: List[int], probs: List[float]) -> None:
    df = pd.DataFrame(
        {
            "id": ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "p_sarcastic": probs,
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
