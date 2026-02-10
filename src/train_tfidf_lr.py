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
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["text", "demojized", "full"], default="text")
    p.add_argument("--run_name", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--max_features", type=int, default=50000)
    p.add_argument("--use_char_ngrams", action="store_true")
    p.add_argument("--C", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_splits(data_dir=args.data_dir)

    x_train = get_variant_column(splits.train, args.variant).astype(str).tolist()
    y_train = splits.train["sarcastic"].astype(int).to_numpy()

    x_dev = get_variant_column(splits.dev, args.variant).astype(str).tolist()
    y_dev = splits.dev["sarcastic"].astype(int).to_numpy()

    if args.use_char_ngrams:
        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=args.max_features,
        )
    else:
        vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=args.max_features,
        )

    Xtr = vec.fit_transform(x_train)
    Xdv = vec.transform(x_dev)

    clf = LogisticRegression(
        C=args.C,
        max_iter=2000,
        class_weight="balanced",
    )
    clf.fit(Xtr, y_train)

    probs = clf.predict_proba(Xdv)[:, 1]
    y_pred = (probs >= 0.5).astype(int)

    m = compute_metrics(y_dev, y_pred)

    out_dir = ensure_dir(Path("results") / args.run_name)
    save_json(out_dir / "metrics.json", {"accuracy": m.accuracy, "macro_f1": m.macro_f1, "per_class": m.per_class})
    save_json(out_dir / "confusion.json", {"labels": [0, 1], "matrix": m.confusion})

    ids = splits.dev.index.astype(int).tolist()
    save_preds_csv(out_dir / "preds.csv", ids, y_dev.tolist(), y_pred.tolist(), probs.tolist())

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

    print(f"[{args.run_name}] dev macro_f1={m.macro_f1:.4f} acc={m.accuracy:.4f}")


if __name__ == "__main__":
    main()
