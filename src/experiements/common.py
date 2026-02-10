from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class SplitData:
    train: pd.DataFrame
    dev: pd.DataFrame


def load_splits(
    data_dir: str | Path = "data",
    train_file: str = "train.En.csv",
    splits_file: str = "splits.json",
) -> SplitData:
    data_dir = Path(data_dir)

    df = pd.read_csv(data_dir / train_file)
    with (data_dir / splits_file).open("r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids = set(splits["train_ids"])
    dev_ids = set(splits["dev_ids"])

    df_train = df[df.index.isin(train_ids)].copy()
    df_dev = df[df.index.isin(dev_ids)].copy()

    return SplitData(train=df_train, dev=df_dev)


def get_variant_column(df: pd.DataFrame, variant: str) -> pd.Series:
    if variant == "full":
        return df["tweet"]
    if variant == "text" and "text_only" in df.columns:
        return df["text_only"]
    if variant == "demojized" and "emoji_to_text" in df.columns:
        return df["emoji_to_text"]
    if variant == "emoji" and "emoji_only" in df.columns:
        return df["emoji_only"]

    return df["tweet"]
