Download train.En.csv and task_A_En_test.csv from:

https://github.com/iabufarha/iSarcasmEval

## Preprocessing and Data Splits

- Emoji-aware preprocessing is implemented in `src/data/preprocess.py`.
- The script `data/prepare_splits.py` generates stratified train/dev splits from `train.En.csv`.
- Split indices are saved in `data/splits.json` for reproducibility.
- The official iSarcasmEval Task A English test file (`task_A_En_test.csv`) is kept untouched.
- Reusable data loading utilities are provided in `src/data/loader.py`.

