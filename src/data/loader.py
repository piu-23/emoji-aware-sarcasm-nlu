import json
import pandas as pd
from .preprocess import add_variants


def load_train_dev(
    train_csv_path: str = "data/train.En.csv",
    splits_path: str = "data/splits.json",
):
    df = pd.read_csv(train_csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = add_variants(df)
    df = df.reset_index(drop=True)
    df["id"] = df.index

    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_df = df[df["id"].isin(splits["train_ids"])].copy()
    dev_df = df[df["id"].isin(splits["dev_ids"])].copy()

    return train_df, dev_df


def load_test(test_csv_path: str = "data/task_A_En_test.csv"):
    df = pd.read_csv(test_csv_path)
    df = add_variants(df)
    return df
