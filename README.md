# Emoji-Aware Sarcasm Detection

This repository contains the implementation for the final project **“Emoji-Aware Sarcasm Detection”**, which studies how emojis contribute to understanding **implicit meaning and sarcasm** in short social media texts.

The project focuses on **binary sarcasm detection** and proposes an **emoji-aware gated fusion Transformer model**, evaluated against strong classical and neural baselines.

---

## 1. Motivation

Sarcasm is a form of **non-literal language**, where the intended meaning often differs from the surface text.  
In social media, emojis frequently act as **pragmatic cues** that help humans infer sarcasm:

- *“Great job 🙂”* → genuine praise  
- *“Great job 🙄”* → sarcastic criticism  

While humans interpret this easily, NLP models often struggle.  
This project investigates whether and how **explicit emoji modeling** improves sarcasm detection.

---

## 2. Task Definition

**Task:** Binary sarcasm detection

- **Input:** A short text (tweet) that may contain emojis  
- **Output:** `sarcastic ∈ {0, 1}`  
  - `0`: Not Sarcastic  
  - `1`: Sarcastic  

---

## 3. Dataset

### Main Dataset
- **iSarcasmEval (English)**

**Used fields:**
- `tweet`: input text  
- `sarcastic`: binary target label  
- `rephrase`: non-sarcastic rewording of sarcastic tweets (used for evaluation)  

Additional fine-grained labels (irony, satire, rhetorical question, etc.) are **not used for training**, but may be analyzed qualitatively.

---

## 4. Data Splits

- **Train:** model learning  
- **Dev:** hyperparameter tuning and model selection  
- **Test:** final evaluation (used once)

If an official test set is provided, it is kept untouched.  
Otherwise, we use a **stratified 80/10/10 split**.

---

## 5. Preprocessing

We apply **minimal preprocessing** to preserve sarcasm cues:

- Normalize whitespace
- Preserve punctuation and casing
- Optional normalization of URLs and user mentions

### Emoji Processing
For each tweet, we construct multiple representations:

1. **Text-only** (`x_text`): emojis removed  
2. **Text + emoji** (`x_full`): original tweet  
3. **Emoji → text** (`x_desc`): emojis replaced with textual descriptions  
4. **Emoji-only** (`x_emoji`): only emojis extracted  

These variants are used across baselines and models.

---

## 6. Models

### 6.1 Baseline 1: TF-IDF + Logistic Regression
- Word n-grams (1–2)
- Optional character n-grams (3–5)
- Class-weighted Logistic Regression
- Input variants:
  - Text-only
  - Emoji → text

### 6.2 Baseline 2: Transformer Classifier
- `roberta-base` (default) or `bert-base-uncased`
- Fine-tuned with a classification head
- Input variants:
  - Text-only
  - Text + emoji
  - Emoji → text

These baselines answer:
- *Do emojis help at all?*
- *How should emojis be represented?*

---

## 6.3 Proposed Model: Emoji-Aware Gated Fusion Transformer

### Architecture Overview
The model builds **two parallel representations**:

1. **Text stream**
   - Input: `x_text`
   - Encoder: Transformer
   - Output: `h_text`

2. **Emoji stream**
   - Input: `x_emoji`
   - Encoder: Transformer (shared weights)
   - Output: `h_emoji`
   - If no emojis: zero vector or `<NO_EMOJI>` embedding

### Gated Fusion
A learnable gate decides how much to rely on text vs emojis:
g = sigmoid(W · [h_text ; h_emoji] + b)
h = g ⊙ h_text + (1 − g) ⊙ h_emoji

The fused representation `h` is passed to a classifier.

**Key idea:**  
The model learns *when emojis matter* and *when they do not*.

---

## 7. Training Setup

- Max sequence length: 128  
- Optimizer: AdamW  
- Learning rate: {1e-5, 2e-5, 3e-5}  
- Batch size: 8–16  
- Epochs: 3–6 with early stopping  
- Primary selection metric: **Macro-F1**  
- Optional: class-weighted loss for imbalance  

---

## 8. Evaluation

### 8.1 Standard Metrics
Reported on the test set:
- Accuracy
- **Macro-F1 (primary)**
- Precision / Recall / F1 per class
- Confusion matrix

### 8.2 Emoji-Subset Evaluation
To isolate emoji effects:
- Evaluate only on examples containing emojis

### 8.3 Minimal-Pair Evaluation (Mandatory)
Using `(tweet, rephrase)` pairs:
- Compute `P(sarcastic | tweet)` vs `P(sarcastic | rephrase)`
- Report:
  - **Pair Preference Accuracy**
  - Average probability margin

A good model should assign higher sarcasm probability to the sarcastic version.

---

## 9. Ablation Studies

Run on the dev set:
1. Remove emoji stream  
2. Remove gate (simple concatenation)  
3. Emoji-only model (sanity check)

These verify that improvements come from the intended design.

---

## 10. Error Analysis

- Sample 50–100 misclassified examples
- Categorize errors (e.g., emoji ambiguity, missing context, rhetorical questions)
- Qualitative comparison:
  - Cases where emoji-aware model succeeds
  - Cases where it fails

---

## 11. Research Questions

- **RQ1:** Do emojis improve sarcasm detection?
- **RQ2:** Does gated fusion outperform standard transformers?
- **RQ3:** Does the model behave sensibly on controlled paired inputs?

---

## 12. References

- Chauhan et al. (2022). *An emoji-aware multitask framework for multimodal sarcasm detection*.  
- iSarcasmEval Shared Task

---

## 13. Notes

This repository is developed as part of a **Natural Language Understanding course project** and focuses on **pragmatics, implicit meaning, and emoji-aware modeling**.
