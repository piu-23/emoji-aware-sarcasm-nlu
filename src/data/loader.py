import json
import pandas as pd
from .preprocess import add_variants


def load_train_dev(
    train_csv_path: str = "data/train.En.csv",
    splits_path: str = "data/splits.json",
):
    """
    Loads and returns the training and validation datasets.

    Steps performed:
      - Reads the training CSV file
      - Removes accidental index column if present
      - Applies emoji-aware preprocessing (add_variants)
      - Assigns unique row IDs
      - Loads split indices from splits.json
      - Returns filtered train and dev DataFrames

    Returns:
      train_df, dev_df
    """
    df = pd.read_csv(train_csv_path)

    # Remove accidental index column if present
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
    """
    Loads and returns the official test dataset.

    Steps performed:
      - Reads the test CSV file
      - Applies emoji-aware preprocessing (add_variants)

    Returns:
      Preprocessed test DataFrame
    """
    df = pd.read_csv(test_csv_path)
    df = add_variants(df)
    return df
