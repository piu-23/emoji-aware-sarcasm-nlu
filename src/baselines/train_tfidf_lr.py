from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data.loader import load_train_dev
from src.experiments.io import ensure_dir, save_json, save_preds_csv
from src.experiments.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    """Read CLI args for the baseline run."""
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["text", "demojized", "full"], default="text")
    p.add_argument("--run_name", required=True)
    p.add_argument("--train_csv", default="data/train.En.csv")
    p.add_argument("--splits_json", default="data/splits.json")
    p.add_argument("--max_features", type=int, default=50000)
    p.add_argument("--use_char_ngrams", action="store_true")
    p.add_argument("--C", type=float, default=1.0)
    return p.parse_args()


def variant_to_column(variant: str) -> str:
    """Map CLI variant name to the preprocessed dataframe column."""
    if variant == "text":
        return "x_text"
    if variant == "demojized":
        return "x_desc"
    if variant == "full":
        return "x_full"
    raise ValueError(f"Unknown variant: {variant}")


def build_vectorizer(use_char_ngrams: bool, max_features: int) -> TfidfVectorizer:
    """Create TF-IDF vectorizer."""
    if use_char_ngrams:
        return TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=max_features)
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=max_features)


def main() -> None:
    args = parse_args()

    train_df, dev_df = load_train_dev(
        train_csv_path=args.train_csv,
        splits_path=args.splits_json,
    )

    col = variant_to_column(args.variant)

    x_train = train_df[col].astype(str).tolist()
    y_train = train_df["sarcastic"].astype(int).to_numpy()

    x_dev = dev_df[col].astype(str).tolist()
    y_dev = dev_df["sarcastic"].astype(int).to_numpy()

    vec = build_vectorizer(args.use_char_ngrams, args.max_features)
    Xtr = vec.fit_transform(x_train)
    Xdv = vec.transform(x_dev)

    clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, y_train)

    probs = clf.predict_proba(Xdv)[:, 1]
    y_pred = (probs >= 0.5).astype(int)

    m = compute_metrics(y_dev, y_pred)

    out_dir = ensure_dir(Path("results") / args.run_name)
    save_json(
        out_dir / "metrics.json",
        {"accuracy": m.accuracy, "macro_f1": m.macro_f1, "per_class": m.per_class},
    )
    save_json(out_dir / "confusion.json", {"labels": [0, 1], "matrix": m.confusion})

    ids = dev_df["id"].astype(int).tolist() if "id" in dev_df.columns else dev_df.index.astype(int).tolist()
    save_preds_csv(out_dir / "preds.csv", ids, y_dev.tolist(), y_pred.tolist(), probs.tolist())

    save_json(
        out_dir / "config.json",
        {
            "model": "tfidf+lr",
            "variant": args.variant,
            "column": col,
            "max_features": args.max_features,
            "use_char_ngrams": bool(args.use_char_ngrams),
            "C": args.C,
            "train_csv": args.train_csv,
            "splits_json": args.splits_json,
        },
    )

    print(f"[{args.run_name}] dev macro_f1={m.macro_f1:.4f} acc={m.accuracy:.4f}")


if __name__ == "__main__":
    main()
