# Contribution Report — Member B

**Role:** Baselines, Transformer baselines, and training orchestration  
**Project:** Emoji-Aware Sarcasm Detection  
**Responsibility Focus:** Model training, evaluation metrics, and results tables

---

## 1. Responsibilities

I was responsible for implementing and running all **baseline models** and the **standard transformer baselines**, as well as producing the main quantitative results used for comparison with the proposed emoji-aware model.

Specifically, my responsibilities included:
- Implementing classical and neural baseline models
- Designing a clean, configurable training and evaluation pipeline
- Running controlled experiments across multiple input variants
- Producing standardized metrics, confusion matrices, and prediction outputs
- Preparing the main results tables used in analysis and presentation

---

## 2. Baseline Models Implemented

### 2.1 TF-IDF + Logistic Regression Baseline

Implemented a classical NLP baseline using:
- TF-IDF features
- Logistic Regression classifier

**Input variants tested:**
- Text-only (emojis removed)
- Emoji → text (demojized emojis)

**Key design choices:**
- Word n-grams: (1, 2)
- Optional character n-grams: (3, 5) for social media text
- Class-weighted loss to handle imbalance
- Regularization strength `C` tuned on the dev set

**Script:**
- `src/train_baseline_lr.py`

This baseline establishes a strong, interpretable lower bound for sarcasm detection.

---

### 2.2 Transformer Baselines

Implemented standard fine-tuning of transformer encoders with a classification head.

**Model:**
- `roberta-base` (default)

**Input variants tested (separate runs):**
- Text-only
- Text + emoji (original tweet)
- Emoji → text (demojized)

These runs directly address:
- Whether emojis help beyond contextual language models
- Which emoji representation is most effective without architectural changes

**Script:**
- `src/train_transformer.py` (config-driven)

---

## 3. Training Orchestration

I designed the training pipeline to be:
- **Config-driven**
- **Reproducible**
- **Comparable across models**

### Shared training settings
- Max sequence length: 128
- Optimizer: AdamW
- Learning rate grid: {1e-5, 2e-5, 3e-5}
- Weight decay: 0.01
- Batch size: 8–16
- Epochs: 3–6
- Early stopping on **dev Macro-F1**
- Warmup: 10% of training steps

Each run outputs:
- `metrics.json`
- `preds.csv`
- `config.yaml`

This ensured fair and transparent comparison across baselines.

---

## 4. Evaluation Outputs Produced

For each baseline run, I generated:

- Accuracy
- Macro-F1 (primary metric)
- Precision / Recall / F1 per class
- Confusion matrices
- Saved predictions for downstream error analysis

**Output locations:**
- `results/main_table.csv`
- `results/confusions/`
- `results/preds/`

These outputs were later used for:
- Emoji-subset evaluation
- Rephrase-pair evaluation
- Error analysis by other team members

---

## 5. Key Results Ownership

I am the owner of:
- All baseline scores reported in the main results table
- Transformer baseline comparisons (text-only vs emoji-aware inputs)
- Stability checks and re-runs when variance was observed

These results form the quantitative reference point against which the proposed gated fusion model is evaluated.

---

## 6. Key Decisions and Insights

- Used **Macro-F1** as the primary metric due to class imbalance
- Kept transformer baselines architecture-identical across input variants to isolate emoji effects
- Saved predictions systematically to enable later qualitative analysis
- Ensured that all baselines were trained and selected using **dev-only** performance

---

## 7. Evidence of Contribution

Artifacts produced and owned:
- `src/train_baseline_lr.py`
- `src/train_transformer.py`
- `results/main_table.csv`
- `results/confusions/`
- `results/preds/`

These scripts and outputs were used directly in:
- Final evaluation
- Presentation figures
- Report tables

---

## 8. Summary

My contribution ensures that:
- The proposed emoji-aware model is evaluated against **strong, well-tuned baselines**
- Improvements are **quantitatively justified**
- Results are **reproducible and interpretable**

This work directly supports the evaluation rigor and credibility of the project.
