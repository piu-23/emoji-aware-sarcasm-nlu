from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    """Save a dictionary as a formatted JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_preds_csv(
    path: str | Path,
    ids: List[int],
    y_true: List[int],
    y_pred: List[int],
    probs: List[float],
) -> None:
    """Save prediction outputs to CSV."""
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
