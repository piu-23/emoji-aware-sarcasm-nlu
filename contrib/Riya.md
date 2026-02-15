# Riya Arawatagi — Contribution (Data Pipeline)

## Overview
I implemented the data preprocessing and splitting pipeline for the iSarcasmEval sarcasm detection project (Task A, English). My goal was to make the dataset reproducible, emoji-aware, and easy for the training/evaluation code to consume.

## Key Work Completed

### 1) Emoji-aware preprocessing (src/data/preprocess.py)
- Implemented preprocessing utilities to generate multiple input variants per example:
  - `x_full`: original text (with emojis)
  - `x_text`: text-only (emojis removed)
  - `x_desc`: emoji-to-text (demojized tokens)
  - `x_emoji`: emoji-only stream (or `<NO_EMOJI>`)
  - `has_emoji`: boolean indicator
- Handled schema mismatch between files (train uses `tweet`, test uses `text`) using robust column detection.

### 2) Reproducible train/dev splits (data/prepare_splits.py + data/splits.json)
- Created stratified train/dev split from `train.En.csv` using a fixed random seed.
- Saved split indices and configuration in `data/splits.json` (`train_ids`, `dev_ids`, `seed`, `dev_size`).
- Ensured the official Task A English test file remains untouched for final evaluation.

### 3) Data loading utilities (src/data/loader.py)
- Implemented loaders to return preprocessed splits consistently:
  - `load_train_dev()` loads, preprocesses, and filters train/dev using `splits.json`
  - `load_test()` loads and preprocesses the official test file

### 4) Dataset statistics (data/stats.md)
- Generated dataset summary including:
  - number of samples
  - label distribution
  - emoji usage rates
  - split-level sanity checks (e.g., no overlap)

## Files Added / Modified
- `src/data/preprocess.py`
- `src/data/loader.py`
- `src/data/__init__.py`
- `data/prepare_splits.py`
- `data/splits.json`
- `data/stats.md`
- `data/README_data.md` (documentation updates)

