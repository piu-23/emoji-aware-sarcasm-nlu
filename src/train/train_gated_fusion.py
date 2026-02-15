from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_train_dev
from src.models.gated_fusion import GatedFusionClassifier


class SarcasmDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, emoji_col: str):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.emoji_col = emoji_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        return {
            "text": str(r[self.text_col]),
            "emoji": str(r[self.emoji_col]),
            "label": int(r["sarcastic"]),
            "has_emoji": bool(r.get("has_emoji", False)),
            "id": r.get("id", idx),
            "rephrase": r.get("rephrase", None),
        }


def make_collate(tokenizer, max_len: int):
    def collate(batch: List[Dict]):
        texts = [b["text"] for b in batch]
        emojis = [b["emoji"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        has_emoji = torch.tensor([int(b["has_emoji"]) for b in batch], dtype=torch.long)

        text_inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        emoji_inputs = tokenizer(emojis, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

        return text_inputs, emoji_inputs, labels, has_emoji

    return collate


@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs, preds, labels = [], [], []
    has_emoji_all = []

    for text_inputs, emoji_inputs, y, has_emoji in loader:
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        emoji_inputs = {k: v.to(device) for k, v in emoji_inputs.items()}
        y = y.to(device)

        out = model(text_inputs, emoji_inputs)
        p = torch.softmax(out.logits, dim=-1)[:, 1]  # P(sarcastic)
        pr = torch.argmax(out.logits, dim=-1)

        probs.append(p.cpu().numpy())
        preds.append(pr.cpu().numpy())
        labels.append(y.cpu().numpy())
        has_emoji_all.append(has_emoji.cpu().numpy())

    return (
        np.concatenate(probs),
        np.concatenate(preds),
        np.concatenate(labels),
        np.concatenate(has_emoji_all),
    )


def metrics_from_preds(y_true, y_pred) -> Dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_preds_csv(out_dir: Path, df: pd.DataFrame, probs, preds, split_name: str):
    out = df.copy()
    out["p_sarcastic"] = probs
    out["pred"] = preds
    out_path = out_dir / f"preds_{split_name}.csv"
    out.to_csv(out_path, index=False)


def main():
    pretrained = "roberta-base"
    fusion_mode = "gated"  # gated | text_only | concat | emoji_only
    text_col = "x_text"
    emoji_col = "x_emoji"

    max_len = 128
    batch_size = 16
    lr = 2e-5
    epochs = 4
    seed = 42

    run_name = f"{fusion_mode}_roberta_lr{lr}_ep{epochs}_bs{batch_size}_seed{seed}"
    out_dir = REPO_ROOT / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_df, dev_df = load_train_dev()

    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<NO_EMOJI>"]})

    train_ds = SarcasmDataset(train_df, text_col=text_col, emoji_col=emoji_col)
    dev_ds = SarcasmDataset(dev_df, text_col=text_col, emoji_col=emoji_col)

    collate = make_collate(tokenizer, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = GatedFusionClassifier(pretrained_name=pretrained, fusion_mode=fusion_mode)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # class-weighted loss for imbalance
    y = train_df["sarcastic"].astype(int).values
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    w0 = 1.0
    w1 = n0 / max(n1, 1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = -1.0
    best_path = out_dir / "best.pt"

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for text_inputs, emoji_inputs, labels, _ in train_loader:
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            emoji_inputs = {k: v.to(device) for k, v in emoji_inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(text_inputs, emoji_inputs)
            loss = loss_fn(out.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())

        dev_probs, dev_preds, dev_labels, dev_has_emoji = predict(model, dev_loader, device)
        dev_m = metrics_from_preds(dev_labels, dev_preds)

        mask = dev_has_emoji == 1
        if mask.any():
            dev_m_emoji = metrics_from_preds(dev_labels[mask], dev_preds[mask])
        else:
            dev_m_emoji = {"macro_f1": None, "accuracy": None, "confusion_matrix": None}

        print(
            f"Epoch {ep} | train_loss={total_loss/len(train_loader):.4f} "
            f"| dev_f1={dev_m['macro_f1']:.4f} dev_acc={dev_m['accuracy']:.4f} "
            f"| dev_emoji_f1={dev_m_emoji['macro_f1']}"
        )

        if dev_m["macro_f1"] > best_f1:
            best_f1 = dev_m["macro_f1"]
            torch.save(
                {"model_state": model.state_dict(), "pretrained": pretrained, "fusion_mode": fusion_mode},
                best_path,
            )
            save_preds_csv(out_dir, dev_df, dev_probs, dev_preds, split_name="dev")

    run_info = {
        "run_name": run_name,
        "pretrained": pretrained,
        "fusion_mode": fusion_mode,
        "text_col": text_col,
        "emoji_col": emoji_col,
        "max_len": max_len,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "best_dev_macro_f1": best_f1,
        "train_label_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_info, indent=2))
    print("\nSaved run to:", out_dir)


if __name__ == "__main__":
    main()
