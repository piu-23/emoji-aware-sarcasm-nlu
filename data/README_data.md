# Data Instructions

## Dataset Download

This project uses the English subset of the iSarcasmEval dataset.

Download the required files from the official repository:
https://github.com/iabufarha/iSarcasmEval

Required files:
- `train/train.En.csv`
- `test/task_A_En_test.csv`

Place both files inside the `data/` directory before running any scripts.

---

## Preprocessing and Data Pipeline

The raw CSV files are kept unchanged. All additional text representations are generated dynamically during preprocessing.

Emoji-aware preprocessing is implemented in:
src/data/preprocess.py

For each example, the following input variants are created:

- `x_full` — original text (with emojis)
- `x_text` — text-only (emojis removed)
- `x_desc` — emoji converted to text tokens (demojized)
- `x_emoji` — emoji-only stream (or `<NO_EMOJI>`)
- `has_emoji` — boolean flag indicating emoji presence

This design allows evaluation of text-only, emoji-only, and emoji-aware baselines.

---

## Train / Validation Split

The script:
data/prepare_splits.py

- Generates a stratified train/dev split from `train.En.csv`
- Uses a fixed random seed for reproducibility
- Preserves class balance across splits
- Keeps the official test set untouched

Split indices and configuration are saved in:
data/splits.json

---

## Data Loading

Reusable loading utilities are provided in:
src/data/loader.py

Functions:
- `load_train_dev()` — returns preprocessed train and validation sets
- `load_test()` — returns the preprocessed official test set

All preprocessing is applied automatically when loading the data.



