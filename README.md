# CS3AM: Bankruptcy Early Warning

Group coursework for CS3AM (Artificial Intelligence and Machine Learning), University of Reading.

The project tests whether XGBoost and a PyTorch neural network predict corporate bankruptcy one year ahead better than the Altman Z' score. Both models are compared against five scorers in total: original Z', refitted Z', logistic regression on all 64 ratios, XGBoost and an MLP.

**Team:** Abdalla, Cameron and Zack
**Deadline:** Fri 4 Dec 2026, noon (group submission on Thu 2 Dec)

## Dataset

Polish Companies Bankruptcy, UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data

* Five ARFF files in `data/raw/`. The main experiment uses `5year.arff` (5,910 statements, 410 bankrupt within one year).
* File numbering runs opposite to the horizon: `1year.arff` is the 5 year horizon.
* `load.py` removes exact duplicate rows before the split and logs the count.
* Credit: Tomczak, S. (2016), Polish Companies Bankruptcy, UCI Machine Learning Repository, DOI 10.24432/C5F600, CC BY 4.0.

## Setup

Requires Python 3.11. On macOS, XGBoost also needs the OpenMP runtime from Homebrew (`brew install libomp`; on Apple Silicon this must be the native Homebrew in `/opt/homebrew`).

For a new clone, from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements_dev.txt
nbstripout --install             # strips notebook outputs on commit
pytest
```

Nominated machine for freeze, test run and final rerun: Abdalla's MacBook Pro (Apple M2 Pro, macOS 26.6), Python 3.11.15

Seeds, paths, split and cross validation settings live in `config.yaml` and are read through `src/config.py`.

## How to run

Run the notebooks in order:

1. `notebooks/01_eda.ipynb`: exploratory data analysis
2. `notebooks/02_models.ipynb`: training and cross validation for all five scorers
3. `notebooks/03_results.ipynb`: final evaluation, figures and tables; reproduces every reported number from a clean environment

Outputs are written to `results/` (metrics as JSON, figures as PNG).

## Repository layout

```
data/raw/            five ARFF files
data/splits/         frozen test indices (do not edit)
src/config.py        repository root and config.yaml loader
src/load.py          loading, label decoding, duplicate removal
src/split.py         frozen stratified 80/20 split, seed 42
src/preprocess.py    preprocessing pipeline
src/baselines.py     original Z', refitted Z', logistic regression
src/xgb_model.py     XGBoost
src/mlp_model.py     PyTorch MLP
src/evaluate.py      metrics, calibration, bootstrap, hypothesis tests
src/explain.py       SHAP
notebooks/           01_eda, 02_models, 03_results
results/             metrics and figures
tests/               raw count, test indices guard and config checks
docs/decision_log.md decisions and reasons
```

## Rules

1. Only `evaluate.py` reads the test indices. The test set is scored once, on Thu 26 Nov.
2. Every transform, calibrator and cost threshold is fitted on training data only.
3. The five data files are never pooled.
4. All changes reach `main` through a pull request reviewed by Abdalla (or Cameron as deputy).
5. Branches are named `owner/topic`. Borrowed code carries a source comment.