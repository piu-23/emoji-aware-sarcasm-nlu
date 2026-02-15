"""
Smoke test for data pipeline.

Run from repo root:
    python src/train/smoke_test_loader.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# Add repo root to Python path so "import src...." works
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from src.data.loader import load_train_dev  # now this works

    train_path = REPO_ROOT / "data" / "train.En.csv"
    splits_path = REPO_ROOT / "data" / "splits.json"

    print("Repo root:", REPO_ROOT)
    print("train.En.csv exists:", train_path.exists(), train_path)
    print("splits.json exists:", splits_path.exists(), splits_path)

    if not train_path.exists():
        print("\n❌ Missing data/train.En.csv. Download iSarcasmEval English train and place it there.")
        return

    # Load splits + variants
    train_df, dev_df = load_train_dev()
    print("\n✅ Loaded dataframes")
    print("Train size:", len(train_df))
    print("Dev size:", len(dev_df))

    print("\nColumns:", train_df.columns.tolist())

    # Show one example row with key fields
    row = train_df.iloc[0]
    print("\nExample row preview:")
    for k in ["tweet", "sarcastic", "x_text", "x_emoji", "x_desc", "has_emoji", "rephrase"]:
        if k in train_df.columns:
            val = row[k]
            if isinstance(val, str):
                val = val.replace("\n", " ")
                val = val[:120]
            print(f"  {k}: {val}")

    # Emoji subset quick check
    if "has_emoji" in train_df.columns:
        emoji_rate = train_df["has_emoji"].mean()
        print(f"\nTrain emoji rate: {emoji_rate:.3f}")

    # sanity: label distribution
    if "sarcastic" in train_df.columns:
        print("\nTrain label counts:")
        print(train_df["sarcastic"].value_counts(dropna=False))

    print("\n✅ Smoke test complete.")


if __name__ == "__main__":
    main()

