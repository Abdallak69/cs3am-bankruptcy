"""Shared evaluation harness that every scorer passes through.

This version handles out of fold predictions only. It checks them against the
shared folds from cv_folds, saves and loads them, and computes cross
validation metrics. It reads no test indices.

Later work adds stratified paired bootstrap with shared resample indices, the
H1a to H1c comparisons, H2 verdicts, the H3 horizon test, DeLong test, out of
fold cost thresholds and the test run itself. This is the only module that may
read the test indices; the test set is scored once, after the model freeze.

Prediction format: one DataFrame per file and scorer, with one row per scored
row in each of the 15 folds (so each training row appears once per repeat),
and these columns:

    row_id  row_id of the statement, as in load_training_set
    label   class label, 1 = bankrupt
    scorer  one of SCORERS, the same on every row
    repeat  cross validation repeat from cv_folds, from 0
    fold    fold within the repeat from cv_folds, from 0
    score   uncalibrated risk score, higher meaning more likely bankrupt
            (for original Z' this is -Z')

Owner: Abdalla (changes need Abdalla; Cameron as authorised deputy).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import REPO_ROOT, load_config
from src.load import LABEL
from src.split import cv_folds, load_training_set

SCORERS: tuple[str, ...] = (
    "original_z",
    "refitted_z",
    "logistic",
    "xgboost",
    "mlp",
)
PREDICTION_COLUMNS: list[str] = ["row_id", "label", "scorer", "repeat", "fold", "score"]
FOLD_METRICS: list[str] = ["pr_auc", "roc_auc", "pr_auc_over_share"]


def validate_oof_predictions(
    predictions: pd.DataFrame, stem: str, config: dict[str, Any] | None = None
) -> None:
    """Check that out of fold predictions follow the prediction format.

    Every fold must hold exactly the rows cv_folds scores in it, so the
    predictions of all five scorers are paired row by row.

    Args:
        predictions: Out of fold predictions of one scorer on one file, in
            the prediction format described in the module docstring.
        stem: File name without extension, e.g. 5year.
        config: Parsed project configuration. Read from config.yaml if None.

    Raises:
        ValueError: If the columns differ from PREDICTION_COLUMNS, the frame
            does not hold exactly one scorer from SCORERS, a score is not
            finite, a fold's row_ids differ from those cv_folds scores in it,
            a label differs from load_training_set, or a row lies outside the
            folds from cv_folds.
    """
    if config is None:
        config = load_config()
    if list(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError(
            f"{stem}: expected columns {PREDICTION_COLUMNS}, "
            f"found {list(predictions.columns)}"
        )

    scorers = predictions["scorer"].unique()
    if len(scorers) != 1:
        raise ValueError(f"{stem}: expected one scorer per frame, found {scorers}")
    scorer = scorers[0]
    if scorer not in SCORERS:
        raise ValueError(
            f"{stem}: unknown scorer {scorer!r}, expected one of {SCORERS}"
        )

    if not np.isfinite(predictions["score"].to_numpy(dtype=float)).all():
        raise ValueError(f"{stem} {scorer}: every score must be finite")

    training = load_training_set(stem, config)
    folds = cv_folds(stem, config)
    for repeat, fold, _, score_row_ids in folds:
        part = predictions[
            (predictions["repeat"] == repeat) & (predictions["fold"] == fold)
        ]
        if not pd.Index(part["row_id"]).sort_values().equals(score_row_ids):
            raise ValueError(
                f"{stem} {scorer}: repeat {repeat} fold {fold} must hold exactly "
                "the rows cv_folds scores in that fold"
            )
        expected = training.loc[part["row_id"], LABEL].to_numpy()
        if not np.array_equal(part["label"].to_numpy(), expected):
            raise ValueError(
                f"{stem} {scorer}: repeat {repeat} fold {fold} has a label that "
                "differs from load_training_set"
            )

    # Every fold matched above, so any further rows lie outside the folds.
    if len(predictions) != sum(len(score) for _, _, _, score in folds):
        raise ValueError(f"{stem} {scorer}: rows found outside the cv_folds folds")


def oof_predictions_path(stem: str, scorer: str, config: dict[str, Any]) -> Path:
    """Locate the out of fold predictions file of one file and scorer.

    Args:
        stem: File name without extension, e.g. 5year.
        scorer: One of SCORERS.
        config: Parsed project configuration.

    Returns:
        The path of the CSV holding the predictions.
    """
    pattern = config["paths"]["oof_predictions_pattern"]
    return REPO_ROOT / pattern.format(stem=stem, scorer=scorer)


def save_oof_predictions(
    predictions: pd.DataFrame, stem: str, config: dict[str, Any] | None = None
) -> Path:
    """Validate out of fold predictions and write them to their CSV.

    An existing file is overwritten, because predictions are regenerated
    while a scorer is tuned.

    Args:
        predictions: Out of fold predictions of one scorer on one file, in
            the prediction format.
        stem: File name without extension, e.g. 5year.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        The file written.

    Raises:
        ValueError: If the predictions fail validate_oof_predictions.
    """
    if config is None:
        config = load_config()
    validate_oof_predictions(predictions, stem, config)
    path = oof_predictions_path(stem, predictions["scorer"].iloc[0], config)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return path


def load_oof_predictions(
    stem: str, scorer: str, config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Read the saved out of fold predictions of one file and scorer.

    Args:
        stem: File name without extension, e.g. 5year.
        scorer: One of SCORERS.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        The predictions in the prediction format, validated.

    Raises:
        ValueError: If the file fails validate_oof_predictions or holds a
            different scorer.
    """
    if config is None:
        config = load_config()
    predictions = pd.read_csv(oof_predictions_path(stem, scorer, config))
    validate_oof_predictions(predictions, stem, config)
    if predictions["scorer"].iloc[0] != scorer:
        raise ValueError(f"{stem}: file for {scorer} holds another scorer")
    return predictions


def cv_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    """Compute the rank metrics of each fold and their mean and spread.

    PR AUC is average precision (sklearn average_precision_score), which
    sums precision over each step in recall rather than interpolating
    between points, so it does not overstate the area. Dividing it by the
    fold's bankrupt share gives how many times better it is than a random
    ranking, which can be compared across files of different prevalence.

    Args:
        predictions: Validated out of fold predictions of one scorer, in the
            prediction format.

    Returns:
        A JSON ready dictionary with the scorer name; folds, one entry per
        (repeat, fold) in order with its pr_auc, roc_auc, bankrupt_share and
        pr_auc_over_share; and mean and std, the mean and sample standard
        deviation of pr_auc, roc_auc and pr_auc_over_share over the folds.
    """
    folds = []
    for (repeat, fold), part in predictions.groupby(["repeat", "fold"]):
        pr_auc = average_precision_score(part["label"], part["score"])
        share = part["label"].mean()
        folds.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "pr_auc": float(pr_auc),
                "roc_auc": float(roc_auc_score(part["label"], part["score"])),
                "bankrupt_share": float(share),
                "pr_auc_over_share": float(pr_auc / share),
            }
        )

    table = pd.DataFrame(folds)[FOLD_METRICS]
    return {
        "scorer": str(predictions["scorer"].iloc[0]),
        "folds": folds,
        "mean": table.mean().to_dict(),
        "std": table.std(ddof=1).to_dict(),
    }


def write_metrics_json(
    metrics: dict[str, Any], name: str, config: dict[str, Any] | None = None
) -> Path:
    """Write a metrics dictionary to a JSON file in the metrics folder.

    Keys are sorted and indented so that reruns give identical files and
    diffs stay readable. An existing file of the same name is overwritten.

    Args:
        metrics: JSON ready dictionary, e.g. from cv_metrics.
        name: File name without extension, e.g. 5year_logistic_cv.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        The file written.
    """
    if config is None:
        config = load_config()
    path = REPO_ROOT / config["paths"]["metrics_dir"] / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
