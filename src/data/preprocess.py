import emoji
import pandas as pd


def extract_emojis(text: str) -> str:
    """
    Extracts and returns only the emoji characters from a text string.
    If the input is not a string, returns an empty string.
    """
    if not isinstance(text, str):
        return ""
    return "".join(ch for ch in text if ch in emoji.EMOJI_DATA)


def remove_emojis(text: str) -> str:
    """
    Removes all emoji characters from the text.
    Returns plain text without emojis.
    """
    if not isinstance(text, str):
        return ""
    return emoji.replace_emoji(text, replace="")


def demojize_text(text: str) -> str:
    """
    Converts emojis into textual descriptions.
    """
    if not isinstance(text, str):
        return ""
    return emoji.demojize(text, delimiters=(" ", " "))


def add_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds multiple emoji-aware input representations to the dataframe.

    New columns created:
      - x_full: original text (with emojis)
      - x_text: text-only version (emojis removed)
      - x_desc: emoji converted into text tokens
      - x_emoji: emoji-only stream (or '<NO_EMOJI>' if none)
      - has_emoji: boolean flag indicating emoji presence

    Automatically handles dataset schema differences:
    - Training file uses 'tweet'
    - Test file may use 'text'
    """
    df = df.copy()

    if "tweet" in df.columns:
        text_col = "tweet"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise ValueError("No text column found. Expected 'tweet' or 'text'.")

    df["x_full"] = df[text_col].astype(str)
    df["x_emoji"] = df[text_col].astype(str).apply(extract_emojis)
    df["has_emoji"] = df["x_emoji"].str.len() > 0
    df["x_text"] = df[text_col].astype(str).apply(remove_emojis).str.strip()
    df["x_desc"] = df[text_col].astype(str).apply(demojize_text).str.strip()
    df.loc[~df["has_emoji"], "x_emoji"] = "<NO_EMOJI>"

    return df


