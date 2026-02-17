# Emoji-Aware Sarcasm Detection

This project studies how emojis influence sarcasm detection in short social media texts.

For example:

- “Great job 🙂” → genuine praise  
- “Great job 🙄” → sarcastic criticism  

Our goal is to understand whether explicitly modeling emojis improves sarcasm detection.

We treat this as a **binary classification task**:

- `0` → Not Sarcastic  
- `1` → Sarcastic  

We compare strong baselines with a proposed **emoji-aware gated fusion model built on RoBERTa**.

---

# Team Responsibilities

## Riya
- Designed and implemented the complete data pipeline  
- Built preprocessing and created input variants:
  - text-only  
  - full (text + emoji)  
  - demojized (emoji → text)  
  - emoji-only  

## Aswathy
- Trained all models:
  - TF-IDF + Logistic Regression  
  - RoBERTa baselines  
  - Gated Fusion model  
- Conducted full training experiments  
- Compared variants (text-only, full, demojized)  
- Evaluated results on:
  - Full development set  
  - Emoji-only subset  
- Performed training analysis and model comparison  

## Pranjaly
- Integrated the training pipeline  
- Implemented testing scripts  
- Conducted ablation experiments  
- Performed detailed error analysis  

---

# Project Structure

```
configs/
data/
results/
src/
  ├── data/
  ├── models/
  ├── train/
  ├── eval/
README.md
```

All model outputs are saved inside:

```
results/<run_name>/
```

Each run folder contains:

- `metrics.json`  
- `confusion.json`  
- `preds_dev.csv`  
- `preds_test.csv`  
- `run_config.json`  
- `best.pt`  

---

# Setup and Installation

## 1. Clone the repository

```bash
git clone <repo_url>
cd emoji-aware-sarcasm-detection
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Main libraries used:

- torch  
- transformers  
- scikit-learn  
- pandas  
- numpy  

---

# How to Run the Project

## Step 1 — Run Data Pipeline

```bash
python -m src.data.preprocess
```

This creates the required input variants:

- `x_text`
- `x_full`
- `x_demojized`
- `x_emoji`

---

## Step 2 — Run TF-IDF + Logistic Regression Baselines

```bash
python -m src.baselines.train_tfidf_lr --variant text --run_name lr_text
python -m src.baselines.train_tfidf_lr --variant demojized --run_name lr_demojized
```

---

## Step 3 — Run RoBERTa Baselines

We evaluate three input cases:

- text-only  
- full (text + emoji)  
- demojized  

```bash
python -m src.transformers.train_transformer --config configs/transformer_text_only.json
python -m src.transformers.train_transformer --config configs/transformer_full.json
python -m src.transformers.train_transformer --config configs/transformer_demojized.json
```

---

## Step 4 — Run Gated Fusion Model

This is our main proposed model.

```bash
python -m src.train.train_gated_fusion
```

This script:

- Trains the model  
- Saves the best checkpoint  
- Saves development predictions  
- Stores training metrics  

---

## Step 5 — Testing

```bash
python -m src.eval.test_gated_fusion
```

After training, this:

- Loads the best checkpoint  
- Evaluates on the test set  
- Saves:
  - `test_metrics.json`
  - `preds_test.csv`

---

# Models and Variants Evaluated

We tested the following input variants:

1. Text-only  
2. Full (Text + Emoji)  
3. Demojized (Emoji → Text description)  

Each model was evaluated on:

- Full development set  
- Emoji-only subset  

---

# Evaluation Metrics

We report:

- Accuracy  
- Macro-F1 (primary metric)  
- Confusion matrix  
- Per-class precision / recall / F1  

Macro-F1 is used because the dataset is imbalanced.

---

# Key Observations

- Emojis influence sarcasm detection in specific contexts.  
- Full RoBERTa input is a strong baseline.  
- Gated fusion helps interpret emoji-heavy tweets.  
- Performance difference is clearer on emoji-only subset.  

---

# Final Note

This project was developed for a Natural Language Understanding course.  
It explores sarcasm as a form of pragmatic meaning and studies how emoji-aware modeling improves implicit meaning understanding.
