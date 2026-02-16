import json
import pandas as pd
import emoji
from sklearn.model_selection import train_test_split

SEED = 42
DEV_SIZE = 0.15

TRAIN_PATH = "data/train.En.csv"
TEST_FILE = "task_A_En_test.csv"


def extract_emojis(text):
    """
    Extracts only emoji characters from a given text string.
    Returns a string containing emojis only.
    """
    if not isinstance(text, str):
        return ""
    return "".join(ch for ch in text if ch in emoji.EMOJI_DATA)


def remove_emojis(text):
    """
    Removes all emoji characters from the text.
    Returns plain text without emojis.
    """
    if not isinstance(text, str):
        return ""
    return emoji.replace_emoji(text, replace="")


def demojize_text(text):
    """
    Converts emojis into textual descriptions.
    """
    if not isinstance(text, str):
        return ""
    return emoji.demojize(text, delimiters=(" ", " "))


def add_variants(df):
    """
    Adds multiple emoji-aware input representations to the dataframe:
      - x_full: original text
      - x_text: text-only (emojis removed)
      - x_desc: emoji converted to text tokens
      - x_emoji: emoji-only stream (or <NO_EMOJI>)
      - has_emoji: boolean indicator
    Handles both 'tweet' and 'text' column formats.
    """
    if "tweet" in df.columns:
        text_col = "tweet"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise ValueError("No text column found")

    df = df.copy()
    df["x_full"] = df[text_col].astype(str)
    df["x_emoji"] = df[text_col].astype(str).apply(extract_emojis)
    df["has_emoji"] = df["x_emoji"].str.len() > 0
    df["x_text"] = df[text_col].astype(str).apply(remove_emojis).str.strip()
    df["x_desc"] = df[text_col].astype(str).apply(demojize_text).str.strip()
    df.loc[~df["has_emoji"], "x_emoji"] = "<NO_EMOJI>"

    return df


def main():
    """
    Generates a reproducible stratified train/dev split from train.En.csv.
    Saves split indices and configuration to data/splits.json.
    """
    df = pd.read_csv(TRAIN_PATH)

    # Remove accidental index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = add_variants(df)
    df = df.reset_index(drop=True)
    df["id"] = df.index

    # Split data while keeping label balance
    groups = df[["id", "sarcastic"]]

    train_g, dev_g = train_test_split(
        groups,
        test_size=DEV_SIZE,
        stratify=groups["sarcastic"],
        random_state=SEED
    )

    splits = {
        "train_ids": train_g["id"].tolist(),
        "dev_ids": dev_g["id"].tolist(),
        "test_file": TEST_FILE,
        "seed": SEED,
        "dev_size": DEV_SIZE
    }

    with open("data/splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print("splits.json written successfully")


if __name__ == "__main__":
    main()

