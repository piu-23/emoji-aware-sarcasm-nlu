import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import load_test, load_train_dev
from src.models.gated_fusion import GatedFusionClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


class EvalDataset(Dataset):
    def __init__(self, df, text_col):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "text": str(row[self.text_col]),
            "emoji": str(row["x_emoji"]),
            "y": int(row["sarcastic"]),
            "has_emoji": int(row.get("has_emoji", 0)),
        }


def make_collate(tokenizer, max_len):
    def collate(batch):
        texts = []
        emojis = []
        ys = []
        has_emoji = []

        for b in batch:
            texts.append(b["text"])
            emojis.append(b["emoji"])
            ys.append(b["y"])
            has_emoji.append(b["has_emoji"])

        y = torch.tensor(ys, dtype=torch.long)
        has_emoji = torch.tensor(has_emoji, dtype=torch.long)

        text_inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        emoji_inputs = tokenizer(
            emojis,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        return text_inputs, emoji_inputs, y, has_emoji

    return collate


def load_model_and_tokenizer(run_dir):
    ckpt_path = run_dir / "best.pt"
    cfg_path = run_dir / "run_config.json"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = {}
    if cfg_path.exists():
        cfg = read_json(cfg_path)

    pretrained = ckpt.get("pretrained", cfg.get("pretrained", "roberta-base"))
    fusion_mode = ckpt.get("fusion_mode", cfg.get("fusion_mode", "text_only"))

    tok_dir = run_dir / "tokenizer"
    if tok_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained)

    tokenizer.add_special_tokens({"additional_special_tokens": ["<NO_EMOJI>"]})

    model = GatedFusionClassifier(pretrained_name=pretrained, fusion_mode=fusion_mode)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(ckpt["model_state"])

    final_cfg = dict(cfg)
    final_cfg["pretrained"] = pretrained
    final_cfg["fusion_mode"] = fusion_mode

    return model, tokenizer, final_cfg


@torch.no_grad()
def predict_probs_preds(model, loader, device):
    model.eval()

    probs_list = []
    preds_list = []
    y_list = []
    has_emoji_list = []

    for text_inputs, emoji_inputs, y, he in loader:
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        emoji_inputs = {k: v.to(device) for k, v in emoji_inputs.items()}

        out = model(text_inputs, emoji_inputs)

        probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
        preds = torch.argmax(out.logits, dim=-1).cpu().numpy()

        probs_list.append(probs)
        preds_list.append(preds)
        y_list.append(y.numpy())
        has_emoji_list.append(he.numpy())

    probs = np.concatenate(probs_list)
    preds = np.concatenate(preds_list)
    ys = np.concatenate(y_list)
    has_emoji = np.concatenate(has_emoji_list)

    return probs, preds, ys, has_emoji


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def eval_test_for_run(run_dir, force=False):
    test_metrics_path = run_dir / "test_metrics.json"
    preds_test_path = run_dir / "preds_test.csv"

    if (not force) and test_metrics_path.exists() and preds_test_path.exists():
        return read_json(test_metrics_path)

    model, tokenizer, cfg = load_model_and_tokenizer(run_dir)

    text_col = cfg.get("text_col", "x_text")
    max_len = int(cfg.get("max_len", 128))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_df = load_test()
    if text_col not in test_df.columns:
        raise ValueError(
            f"{run_dir.name}: text_col '{text_col}' not found in test_df columns: {list(test_df.columns)}"
        )

    ds = EvalDataset(test_df, text_col=text_col)
    dl = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        collate_fn=make_collate(tokenizer, max_len=max_len),
    )

    probs, preds, y, has_emoji = predict_probs_preds(model, dl, device)

    m_all = compute_metrics(y, preds)

    mask_emoji = has_emoji == 1
    mask_no = has_emoji == 0

    if mask_emoji.any():
        m_emoji = compute_metrics(y[mask_emoji], preds[mask_emoji])
    else:
        m_emoji = None

    if mask_no.any():
        m_no = compute_metrics(y[mask_no], preds[mask_no])
    else:
        m_no = None

    out = {
        "run_folder": run_dir.name,
        "fusion_mode": cfg.get("fusion_mode", ""),
        "pretrained": cfg.get("pretrained", ""),
        "text_col_used": text_col,
        "max_len": max_len,
        "test_macro_f1": m_all["macro_f1"],
        "test_accuracy": m_all["accuracy"],
        "test_confusion_matrix": m_all["confusion_matrix"],
        "test_emoji_subset": m_emoji,
        "test_no_emoji_subset": m_no,
    }

    write_json(test_metrics_path, out)

    out_df = test_df.copy()
    out_df["p_sarcastic"] = probs
    out_df["pred"] = preds
    out_df.to_csv(preds_test_path, index=False)

    return out


def export_error_analysis(run_dir, topk=50):
    preds_path = run_dir / "preds_test.csv"
    if not preds_path.exists():
        return

    df = pd.read_csv(preds_path)

    if "sarcastic" not in df.columns or "pred" not in df.columns:
        return

    df["sarcastic"] = df["sarcastic"].astype(int)
    df["pred"] = df["pred"].astype(int)

    if "has_emoji" in df.columns:
        df["has_emoji"] = df["has_emoji"].astype(int)
    else:
        def has_emoji_fn(s):
            s = str(s).strip()
            if s == "" or s == "<NO_EMOJI>":
                return 0
            return 1

        df["has_emoji"] = df["x_emoji"].astype(str).apply(has_emoji_fn)

    # pick a readable text column
    if "text" in df.columns:
        text_col = "text"
    elif "tweet" in df.columns:
        text_col = "tweet"
    elif "x_full" in df.columns:
        text_col = "x_full"
    else:
        text_col = "x_text"

    fp = df[(df["sarcastic"] == 0) & (df["pred"] == 1)].copy()
    fn = df[(df["sarcastic"] == 1) & (df["pred"] == 0)].copy()

    if "p_sarcastic" in df.columns:
        fp = fp.sort_values("p_sarcastic", ascending=False)
        fn = fn.sort_values("p_sarcastic", ascending=True)

    out_dir = ensure_dir(run_dir / "analysis")

    wanted_cols = [
        text_col, "sarcastic", "pred", "p_sarcastic", "has_emoji",
        "x_text", "x_full", "x_emoji", "rephrase"
    ]
    cols = []
    for c in wanted_cols:
        if c in df.columns:
            cols.append(c)

    fp.head(topk)[cols].to_csv(out_dir / f"false_positives_top{topk}.csv", index=False)
    fn.head(topk)[cols].to_csv(out_dir / f"false_negatives_top{topk}.csv", index=False)

    fp[fp["has_emoji"] == 1].head(25)[cols].to_csv(out_dir / "fp_with_emoji_top25.csv", index=False)
    fn[fn["has_emoji"] == 1].head(25)[cols].to_csv(out_dir / "fn_with_emoji_top25.csv", index=False)

    summary_lines = []
    summary_lines.append(f"RUN: {run_dir.name}")
    summary_lines.append(f"FP count: {len(fp)}")
    summary_lines.append(f"FN count: {len(fn)}")

    fp_rate = float(fp["has_emoji"].mean()) if len(fp) else 0.0
    fn_rate = float(fn["has_emoji"].mean()) if len(fn) else 0.0
    summary_lines.append(f"FP emoji rate: {fp_rate:.3f}")
    summary_lines.append(f"FN emoji rate: {fn_rate:.3f}")

    summary_lines.append("Files written:")
    summary_lines.append(f" - {out_dir / f'false_positives_top{topk}.csv'}")
    summary_lines.append(f" - {out_dir / f'false_negatives_top{topk}.csv'}")
    summary_lines.append(f" - {out_dir / 'fp_with_emoji_top25.csv'}")
    summary_lines.append(f" - {out_dir / 'fn_with_emoji_top25.csv'}")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def plot_confusion(cm, title, out_path):
    arr = np.array(cm, dtype=int)

    fig, ax = plt.subplots()
    ax.imshow(arr)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    ax.set_xlabel("Pred")
    ax.set_ylabel("Gold")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute even if preds_test/test_metrics exist.")
    parser.add_argument("--only_seed", type=int, default=None, help="Evaluate only runs with this seed (reads run_config.json).")
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        raise FileNotFoundError("results/ folder not found. Train first.")

    run_dirs = []
    for d in RESULTS_DIR.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("_"):
            continue
        if not (d / "best.pt").exists():
            continue
        if not (d / "run_config.json").exists():
            continue

        cfg = read_json(d / "run_config.json")
        seed = cfg.get("seed", None)
        if args.only_seed is not None and seed != args.only_seed:
            continue

        run_dirs.append(d)

    run_dirs = sorted(run_dirs, key=lambda x: x.name)
    if not run_dirs:
        print("No runnable checkpoints found in results/.")
        return

    final_dir = ensure_dir(RESULTS_DIR / "_final")
    plots_dir = ensure_dir(final_dir / "plots")

    summary_rows = []
    warnings = []

    for d in run_dirs:
        try:
            tm = eval_test_for_run(d, force=args.force)
            export_error_analysis(d, topk=50)

            cm = tm.get("test_confusion_matrix")
            if cm:
                plot_confusion(cm, f"{d.name} (test)", plots_dir / f"cm_{d.name}.png")

            summary_rows.append({
                "run_folder": d.name,
                "fusion_mode": tm.get("fusion_mode", ""),
                "text_col_used": tm.get("text_col_used", ""),
                "test_macro_f1": tm.get("test_macro_f1", None),
                "test_accuracy": tm.get("test_accuracy", None),
                "emoji_subset_f1": (tm.get("test_emoji_subset") or {}).get("macro_f1", None),
                "no_emoji_subset_f1": (tm.get("test_no_emoji_subset") or {}).get("macro_f1", None),
            })

            f1v = tm.get("test_macro_f1", None)
            accv = tm.get("test_accuracy", None)
            if f1v is not None and accv is not None:
                print(f"Evaluated {d.name}: f1={f1v:.4f} acc={accv:.4f}")
            else:
                print(f"Evaluated {d.name}")

        except Exception as e:
            warnings.append(f"[FAILED] {d.name}: {e}")
            print(f"Failed {d.name}: {e}")

    df = pd.DataFrame(summary_rows)
    if len(df) > 0:
        df = df.sort_values("test_macro_f1", ascending=False)
    df.to_csv(final_dir / "FINAL_SUMMARY.csv", index=False)

    if len(df) > 0:
        # bar plot: test f1
        fig, ax = plt.subplots()
        ax.bar(df["run_folder"], df["test_macro_f1"])
        ax.set_title("Test Macro-F1 by run")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(plots_dir / "test_macro_f1_bar.png", dpi=200)
        plt.close(fig)

        # emoji vs no-emoji subset
        fig, ax = plt.subplots()
        x = np.arange(len(df))
        ax.bar(x - 0.2, df["emoji_subset_f1"], width=0.4, label="emoji subset")
        ax.bar(x + 0.2, df["no_emoji_subset_f1"], width=0.4, label="no-emoji subset")
        ax.set_title("Test Macro-F1: emoji vs no-emoji subsets")
        ax.set_xticks(x)
        ax.set_xticklabels(df["run_folder"], rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "emoji_subset_macro_f1.png", dpi=200)
        plt.close(fig)

    lines = []
    lines.append("FINAL SUMMARY (sorted by test_macro_f1)")
    lines.append(str(df.to_string(index=False)))
    lines.append("")

    if warnings:
        lines.append("WARNINGS:")
        for w in warnings:
            lines.append(w)
    else:
        lines.append("No warnings.")

    (final_dir / "FINAL_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nWrote:")
    print(" -", final_dir / "FINAL_SUMMARY.csv")
    print(" -", final_dir / "FINAL_SUMMARY.txt")
    print(" - plots in:", plots_dir)


if __name__ == "__main__":
    main()
