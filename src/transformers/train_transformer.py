from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.experiments.common import get_variant_column, load_splits
from src.experiments.io import ensure_dir, save_json, save_preds_csv
from src.experiments.metrics import compute_metrics
from src.transformers.hf_dataset import TextDataset


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


@dataclass
class Cfg:
    run_name: str
    model_name: str
    variant: str
    max_len: int
    lr: float
    weight_decay: float
    epochs: int
    batch_size: int
    warmup_ratio: float
    seed: int
    patience: int


def load_cfg(path: str | Path) -> Cfg:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return Cfg(**raw)


@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    model.eval()
    ys, yp, pr, ids = [], [], [], []
    for batch in loader:
        batch_ids = batch.pop("id").cpu().numpy().tolist()
        labels = batch["labels"].cpu().numpy()
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(**batch)
        logits = out.logits.detach().cpu().numpy()
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        pred = (probs >= 0.5).astype(int)

        ys.append(labels)
        yp.append(pred)
        pr.append(probs)
        ids.extend(batch_ids)

    return np.concatenate(ys), np.concatenate(yp), np.concatenate(pr), ids


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_splits()

    x_train = get_variant_column(splits.train, cfg.variant).astype(str).tolist()
    y_train = splits.train["sarcastic"].astype(int).tolist()
    ids_train = splits.train.index.astype(int).tolist()

    x_dev = get_variant_column(splits.dev, cfg.variant).astype(str).tolist()
    y_dev = splits.dev["sarcastic"].astype(int).tolist()
    ids_dev = splits.dev.index.astype(int).tolist()

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    enc_tr = tok(x_train, truncation=True, padding=True, max_length=cfg.max_len)
    enc_dv = tok(x_dev, truncation=True, padding=True, max_length=cfg.max_len)

    ds_tr = TextDataset(enc_tr, y_train, ids_train)
    ds_dv = TextDataset(enc_dv, y_dev, ids_dev)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True)
    dl_dv = DataLoader(ds_dv, batch_size=cfg.batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = len(dl_tr) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    best_f1 = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        for batch in dl_tr:
            batch.pop("id")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

        y_true, y_pred, probs, ids = predict(model, dl_dv, device)
        m = compute_metrics(y_true, y_pred)

        print(f"epoch={epoch+1} dev macro_f1={m.macro_f1:.4f} acc={m.accuracy:.4f}")

        if m.macro_f1 > best_f1 + 1e-6:
            best_f1 = m.macro_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred, probs, ids = predict(model, dl_dv, device)
    m = compute_metrics(y_true, y_pred)

    out_dir = ensure_dir(Path("results") / cfg.run_name)
    save_json(out_dir / "metrics.json", {"accuracy": m.accuracy, "macro_f1": m.macro_f1, "per_class": m.per_class})
    save_json(out_dir / "confusion.json", {"labels": [0, 1], "matrix": m.confusion})
    save_preds_csv(out_dir / "preds.csv", ids, y_true.tolist(), y_pred.tolist(), probs.tolist())

    save_json(out_dir / "config.json", cfg.__dict__)
    print(f"[{cfg.run_name}] best dev macro_f1={m.macro_f1:.4f}")


if __name__ == "__main__":
    main()
