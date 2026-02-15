from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.experiments.common import load_splits
from src.experiments.io import ensure_dir, save_json, save_preds_csv
from src.experiments.metrics import compute_metrics
from src.transformers.gated_fusion import GatedFusionClassifier


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Cfg:
    model_type: str
    pretrained_name: str
    max_len: int
    batch_size: int
    lr: float
    epochs: int
    seed: int
    run_name: str = "gated_fusion"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_cfg(path: str | Path) -> Cfg:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # YAML sometimes parses scientific notation as strings in Colab.
    for k in ("max_len", "batch_size", "epochs", "seed", "patience"):
        if k in raw:
            raw[k] = int(raw[k])
    for k in ("lr", "weight_decay", "warmup_ratio"):
        if k in raw:
            raw[k] = float(raw[k])

    return Cfg(**raw)

class DualDataset(torch.utils.data.Dataset):
    def __init__(self, enc_a, enc_b, labels, ids, emoji_present):
        self.enc_a = enc_a
        self.enc_b = enc_b
        self.labels = labels
        self.ids = ids
        self.emoji_present = emoji_present

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "a_input_ids": torch.tensor(self.enc_a["input_ids"][idx]),
            "a_attention_mask": torch.tensor(self.enc_a["attention_mask"][idx]),
            "b_input_ids": torch.tensor(self.enc_b["input_ids"][idx]),
            "b_attention_mask": torch.tensor(self.enc_b["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "id": torch.tensor(self.ids[idx], dtype=torch.long),
            "emoji_present": torch.tensor(self.emoji_present[idx], dtype=torch.float),
        }
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    model.eval()
    ys, yp, pr, ids = [], [], [], []
    for batch in loader:
        batch_ids = batch.pop("id").cpu().numpy().tolist()
        y = batch.pop("labels").cpu().numpy()

        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).detach().cpu().numpy()

        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        pred = (probs >= 0.5).astype(int)

        ys.append(y)
        yp.append(pred)
        pr.append(probs)
        ids.extend(batch_ids)

    return np.concatenate(ys), np.concatenate(yp), np.concatenate(pr), ids


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_splits()  # should default to train.En.processed.csv from your earlier patch

    df_tr = splits.train
    df_dv = splits.dev

    x_text_tr = df_tr["text_only"].astype(str).tolist()
    x_emoji_tr = df_tr["emoji_only"].astype(str).tolist()
    emoji_present_tr = [1 if s.strip() else 0 for s in x_emoji_tr]
    y_tr = df_tr["sarcastic"].astype(int).tolist()
    ids_tr = df_tr.index.astype(int).tolist()

    x_text_dv = df_dv["text_only"].astype(str).tolist()
    x_emoji_dv = df_dv["emoji_only"].astype(str).tolist()
    emoji_present_dv = [1 if s.strip() else 0 for s in x_emoji_dv]
    y_dv = df_dv["sarcastic"].astype(int).tolist()
    ids_dv = df_dv.index.astype(int).tolist()

    tok = AutoTokenizer.from_pretrained(cfg.pretrained_name, use_fast=True)

    enc_text_tr = tok(x_text_tr, truncation=True, padding=True, max_length=cfg.max_len)
    enc_emoji_tr = tok(x_emoji_tr, truncation=True, padding=True, max_length=cfg.max_len)
    enc_text_dv = tok(x_text_dv, truncation=True, padding=True, max_length=cfg.max_len)
    enc_emoji_dv = tok(x_emoji_dv, truncation=True, padding=True, max_length=cfg.max_len)

    ds_tr = DualDataset(enc_text_tr, enc_emoji_tr, y_tr, ids_tr, emoji_present_tr)
    ds_dv = DualDataset(enc_text_dv, enc_emoji_dv, y_dv, ids_dv, emoji_present_dv)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True)
    dl_dv = DataLoader(ds_dv, batch_size=cfg.batch_size, shuffle=False)

    model = GatedFusionClassifier(cfg.pretrained_name)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = len(dl_tr) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    best_f1 = -1.0
    best_state = None
    bad = 0

    for epoch in range(cfg.epochs):
        model.train()
        for batch in dl_tr:
            batch.pop("id")
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch)
            loss = torch.nn.functional.cross_entropy(logits, labels)

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
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred, probs, ids = predict(model, dl_dv, device)
    m = compute_metrics(y_true, y_pred)

    out_dir = ensure_dir(Path("results") / cfg.run_name)
    save_json(out_dir / "metrics.json", {"accuracy": m.accuracy, "macro_f1": m.macro_f1, "per_class": m.per_class})
    save_json(out_dir / "confusion.json", {"labels": [0, 1], "matrix": m.confusion})
    save_preds_csv(out_dir / "preds.csv", ids, y_true.tolist(), y_pred.tolist(), probs.tolist())
    save_json(out_dir / "config.json", asdict(cfg))

    print(f"[{cfg.run_name}] best dev macro_f1={m.macro_f1:.4f}")


if __name__ == "__main__":
    main()
