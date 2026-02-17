import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_train_dev
from src.models.gated_fusion import GatedFusionClassifier


class SarcasmDataset(Dataset):
    def __init__(self, df, text_col, emoji_col):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.emoji_col = emoji_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "text": str(row[self.text_col]),
            "emoji": str(row[self.emoji_col]),
            "label": int(row["sarcastic"]),
            "has_emoji": int(row.get("has_emoji", 0)),
            "id": row.get("id", idx),
        }


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_collate(tokenizer, max_len):
    def collate(batch):
        texts = [b["text"] for b in batch]
        emojis = [b["emoji"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        has_emoji = torch.tensor([b["has_emoji"] for b in batch], dtype=torch.long)

        text_inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        emoji_inputs = tokenizer(
            emojis, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )

        return text_inputs, emoji_inputs, labels, has_emoji

    return collate


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    all_has_emoji = []

    for text_inputs, emoji_inputs, labels, has_emoji in loader:
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        emoji_inputs = {k: v.to(device) for k, v in emoji_inputs.items()}
        labels = labels.to(device)

        out = model(text_inputs, emoji_inputs)
        probs = torch.softmax(out.logits, dim=-1)[:, 1]
        preds = torch.argmax(out.logits, dim=-1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_has_emoji.append(has_emoji.cpu().numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_has_emoji),
    )


def metrics(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_preds(out_dir, df, probs, preds, split):
    out = df.copy()
    out["p_sarcastic"] = probs
    out["pred"] = preds
    out.to_csv(out_dir / f"preds_{split}.csv", index=False)


def make_run_name(mode, text_col, lr, epochs, batch_size, seed):
    if lr < 1e-3:
        lr_str = f"{lr:.0e}"
    else:
        lr_str = str(lr)
    return f"{mode}_{text_col}_roberta_lr{lr_str}_ep{epochs}_bs{batch_size}_seed{seed}"


def refuse_overwrite(out_dir):
    if out_dir.exists():
        for _ in out_dir.iterdir():
            raise RuntimeError(
                "Refusing to overwrite a non-empty results folder:\n"
                f"  {out_dir}\n"
                "Change settings (run_name) or delete that folder."
            )


def train_one_mode(mode, pretrained, text_col, emoji_col, max_len, batch_size, lr, epochs, seed):
    run_name = make_run_name(mode, text_col, lr, epochs, batch_size, seed)
    out_dir = REPO_ROOT / "results" / run_name

    refuse_overwrite(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nRun:", run_name)
    print("Device:", device)

    train_df, dev_df = load_train_dev()

    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<NO_EMOJI>"]})
    (out_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir / "tokenizer")

    train_ds = SarcasmDataset(train_df, text_col, emoji_col)
    dev_ds = SarcasmDataset(dev_df, text_col, emoji_col)

    collate_fn = make_collate(tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = GatedFusionClassifier(pretrained_name=pretrained, fusion_mode=mode)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)

    y = train_df["sarcastic"].astype(int).values
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    w1 = n0 / max(n1, 1)
    class_weights = torch.tensor([1.0, w1], dtype=torch.float32).to(device)
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

            optimizer.zero_grad(set_to_none=True)

            out = model(text_inputs, emoji_inputs)
            loss = loss_fn(out.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())

        dev_probs, dev_preds, dev_labels, dev_has_emoji = predict(model, dev_loader, device)
        dev_all = metrics(dev_labels, dev_preds)

        mask = dev_has_emoji == 1
        if mask.any():
            dev_emoji = metrics(dev_labels[mask], dev_preds[mask])
        else:
            dev_emoji = {"macro_f1": None, "accuracy": None, "confusion_matrix": None}

        avg_loss = total_loss / max(len(train_loader), 1)
        print(
            f"Epoch {ep} | loss={avg_loss:.4f} | dev_f1={dev_all['macro_f1']:.4f} "
            f"dev_acc={dev_all['accuracy']:.4f} | dev_emoji_f1={dev_emoji['macro_f1']}"
        )

        if dev_all["macro_f1"] > best_f1 + 1e-6:
            best_f1 = dev_all["macro_f1"]

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "pretrained": pretrained,
                    "fusion_mode": mode,
                    "max_len": max_len,
                    "text_col": text_col,
                    "emoji_col": emoji_col,
                },
                best_path,
            )

            save_preds(out_dir, dev_df, dev_probs, dev_preds, "dev")
            (out_dir / "metrics_dev.json").write_text(json.dumps(dev_all, indent=2), encoding="utf-8")

    run_info = {
        "run_name": run_name,
        "pretrained": pretrained,
        "fusion_mode": mode,
        "text_col": text_col,
        "emoji_col": emoji_col,
        "max_len": max_len,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "best_dev_macro_f1": best_f1,
        "train_label_counts": {"0": n0, "1": n1},
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    print("Saved:", out_dir)
    return out_dir


def load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_preds(run_dir):
    return pd.read_csv(run_dir / "preds_dev.csv")


def plot_results(run_dirs, baseline_preference=("text_only", "x_text")):
    plots_dir = REPO_ROOT / "results" / "_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in run_dirs:
        cfg = load_json(d / "run_config.json")
        met = load_json(d / "metrics_dev.json")
        if not cfg or not met:
            continue
        rows.append(
            {
                "run": d.name,
                "fusion_mode": cfg.get("fusion_mode", ""),
                "text_col": cfg.get("text_col", ""),
                "macro_f1": met.get("macro_f1"),
                "accuracy": met.get("accuracy"),
            }
        )

    if not rows:
        print("No metrics found to plot.")
        return

    dfm = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)

    plt.figure()
    plt.bar(dfm["run"], dfm["macro_f1"])
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("Dev macro-F1 by run")
    plt.tight_layout()
    plt.savefig(plots_dir / "dev_macro_f1_by_run.png", dpi=200)
    plt.close()

    base_mode, base_textcol = baseline_preference
    baseline_dir = None
    gated_dir = None

    for d in run_dirs:
        cfg = load_json(d / "run_config.json")
        if not cfg:
            continue
        if cfg.get("fusion_mode") == "gated" and cfg.get("text_col") == "x_text":
            gated_dir = d
        if cfg.get("fusion_mode") == base_mode and cfg.get("text_col") == base_textcol:
            baseline_dir = d

    if gated_dir is None or baseline_dir is None:
        print("Skipping confusion-matrix plots (missing gated or baseline run).")
        return

    base_preds = load_preds(baseline_dir)
    gated_preds = load_preds(gated_dir)

    def plot_cm(preds_df, title, fname):
        y_true = preds_df["sarcastic"].astype(int).to_numpy()
        y_pred = preds_df["pred"].astype(int).to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        fig, ax = plt.subplots()
        disp.plot(ax=ax, values_format="d")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=200)
        plt.close(fig)

    plot_cm(base_preds, f"Confusion matrix: {baseline_dir.name}", "cm_baseline.png")
    plot_cm(gated_preds, f"Confusion matrix: {gated_dir.name}", "cm_gated.png")

    print("Plots saved to:", plots_dir)


def main():
    pretrained = "roberta-base"
    emoji_col = "x_emoji"

    max_len = 128
    batch_size = 16
    lr = 2e-5
    epochs = 4
    seed = 43

    set_seed(seed)

    experiments = [
        {"mode": "text_only", "text_col": "x_text"},
        {"mode": "text_only", "text_col": "x_full"},
        {"mode": "gated", "text_col": "x_text"},
        {"mode": "concat", "text_col": "x_text"},
        {"mode": "emoji_only", "text_col": "x_text"},
    ]

    run_dirs = []
    for ex in experiments:
        run_dir = train_one_mode(
            mode=ex["mode"],
            pretrained=pretrained,
            text_col=ex["text_col"],
            emoji_col=emoji_col,
            max_len=max_len,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            seed=seed,
        )
        run_dirs.append(run_dir)

    plot_results(run_dirs, baseline_preference=("text_only", "x_text"))


if __name__ == "__main__":
    main()

