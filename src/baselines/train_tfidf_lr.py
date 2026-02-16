from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.experiments.common import get_variant_column, load_splits
from src.experiments.io import ensure_dir, save_json, save_preds_csv
from src.experiments.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["text", "demojized", "full"], default="text")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--use_char_ngrams", action="store_true")
    parser.add_argument("--C", type=float, default=1.0)
    return parser.parse_args()


def build_vectorizer(use_char_ngrams: bool, max_features: int) -> TfidfVectorizer:
    """Create TF-IDF vectorizer."""
    if use_char_ngrams:
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=max_features,
        )
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_features,
    )


def train_and_evaluate(args: argparse.Namespace) -> None:
    """Train TF-IDF + Logistic Regression baseline and evaluate on dev set."""
    splits = load_splits(data_dir=args.data_dir)

    x_train = get_variant_column(splits.train, args.variant).astype(str).tolist()
    y_train = splits.train["sarcastic"].astype(int).to_numpy()

    x_dev = get_variant_column(splits.dev, args.variant).astype(str).tolist()
    y_dev = splits.dev["sarcastic"].astype(int).to_numpy()

    vectorizer = build_vectorizer(args.use_char_ngrams, args.max_features)
    Xtr = vectorizer.fit_transform(x_train)
    Xdv = vectorizer.transform(x_dev)

    clf = LogisticRegression(
        C=args.C,
        max_iter=2000,
        class_weight="balanced",
    )
    clf.fit(Xtr, y_train)

    probs = clf.predict_proba(Xdv)[:, 1]
    y_pred = (probs >= 0.5).astype(int)

    metrics = compute_metrics(y_dev, y_pred)

    out_dir = ensure_dir(Path("results") / args.run_name)

    save_json(
        out_dir / "metrics.json",
        {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "per_class": metrics.per_class,
        },
    )

    save_json(
        out_dir / "confusion.json",
        {"labels": [0, 1], "matrix": metrics.confusion},
    )

    ids = splits.dev.index.astype(int).tolist()
    save_preds_csv(
        out_dir / "preds.csv",
        ids,
        y_dev.tolist(),
        y_pred.tolist(),
        probs.tolist(),
    )

    save_json(
        out_dir / "config.json",
        {
            "model": "tfidf+lr",
            "variant": args.variant,
            "max_features": args.max_features,
            "use_char_ngrams": bool(args.use_char_ngrams),
            "C": args.C,
        },
    )

    print(f"[{args.run_name}] dev macro_f1={metrics.macro_f1:.4f} acc={metrics.accuracy:.4f}")


def main() -> None:
    """Entry point."""
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
