"""Platt calibration for all five scorers, and reliability diagrams.

Platt scaling (Platt, 1999) maps a scorer's uncalibrated risk score s to a
probability of bankruptcy p = 1 / (1 + exp(-(a * s + b))), with a and b from
a logistic regression of the label on the score. Every scorer, including the
original Z', is calibrated this way; the method was fixed in advance in the
decision log.

Each calibrator is fitted on the concatenated out of fold predictions from
all three cross validation repeats of the scorer's final configuration, in
the prediction format of evaluate.py; for the original Z' the scores are -Z'
itself. Nothing here reads test data: the calibrators are saved as JSON at
the model freeze and applied to the test scores by evaluate.py.

PR AUC and ROC AUC use uncalibrated scores. Calibration changes only the
probabilities, so it matters for the Brier score, the reliability diagrams
and the cost thresholds.

A calibrator fitted on out of fold predictions and judged on the same
predictions looks better than it is. cross_fitted_probabilities therefore
calibrates each fold with a calibrator fitted on the other folds of the same
repeat, which gives an honest reliability diagram on training data.

Owner: Zack.

Reference: Platt, J. C. (1999), Probabilistic outputs for support vector
machines and comparisons to regularized likelihood methods, Advances in
Large Margin Classifiers, MIT Press, 61 to 74.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.config import REPO_ROOT, load_config
from src.evaluate import SCORERS, load_oof_predictions

# Quantile bins hold equal numbers of rows. With about 7% bankrupt, equal
# width bins above a probability of 0.3 would be nearly empty.
N_BINS: int = 10
FIGURE_DPI: int = 300


@dataclass(frozen=True)
class PlattCalibrator:
    """A fitted Platt calibrator: p = 1 / (1 + exp(-(slope * score + intercept))).

    Attributes:
        scorer: The scorer it calibrates, one of SCORERS.
        slope: Coefficient a on the uncalibrated score.
        intercept: Constant b.
    """

    scorer: str
    slope: float
    intercept: float

    def predict(self, scores: np.ndarray | pd.Series) -> np.ndarray:
        """Turn uncalibrated risk scores into probabilities of bankruptcy.

        Args:
            scores: Uncalibrated risk scores of the same scorer, higher
                meaning more likely bankrupt.

        Returns:
            One probability per score, between 0 and 1.
        """
        logits = self.slope * np.asarray(scores, dtype=float) + self.intercept
        # expit is the logistic function without overflow for large logits.
        return expit(logits)


def fit_platt(
    scores: np.ndarray | pd.Series, labels: np.ndarray | pd.Series, scorer: str
) -> PlattCalibrator:
    """Fit a Platt calibrator by unpenalised logistic regression of label on score.

    Args:
        scores: Uncalibrated risk scores.
        labels: Class labels, 1 = bankrupt.
        scorer: Name of the scorer, stored with the calibrator.

    Returns:
        The fitted calibrator.
    """
    # C = inf turns the penalty off, so a and b are the maximum likelihood
    # fit; scikit-learn 1.8 deprecated penalty=None.
    model = LogisticRegression(C=np.inf)
    model.fit(np.asarray(scores, dtype=float).reshape(-1, 1), np.asarray(labels))
    return PlattCalibrator(
        scorer=scorer,
        slope=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
    )


def fit_from_oof(predictions: pd.DataFrame) -> PlattCalibrator:
    """Fit the calibrator of one scorer on its out of fold predictions.

    Every row of all three repeats is used, so each training row counts
    three times, once per repeat.

    Args:
        predictions: Validated out of fold predictions of one scorer, in the
            prediction format of evaluate.py.

    Returns:
        The fitted calibrator.
    """
    return fit_platt(
        predictions["score"], predictions["label"], predictions["scorer"].iloc[0]
    )


def cross_fitted_probabilities(predictions: pd.DataFrame) -> pd.Series:
    """Calibrate each fold with a calibrator fitted on the other folds of its repeat.

    Used only to judge calibration on training data; the saved calibrators
    come from fit_from_oof.

    Args:
        predictions: Validated out of fold predictions of one scorer.

    Returns:
        One probability per row, indexed like predictions.
    """
    scorer = predictions["scorer"].iloc[0]
    probabilities = pd.Series(np.nan, index=predictions.index, name="probability")
    for (repeat, fold), part in predictions.groupby(["repeat", "fold"]):
        others = predictions[
            (predictions["repeat"] == repeat) & (predictions["fold"] != fold)
        ]
        calibrator = fit_platt(others["score"], others["label"], scorer)
        probabilities.loc[part.index] = calibrator.predict(part["score"])
    return probabilities


def brier_score(
    labels: np.ndarray | pd.Series, probabilities: np.ndarray | pd.Series
) -> float:
    """Compute the Brier score, the mean squared error of the probabilities.

    Args:
        labels: Class labels, 1 = bankrupt.
        probabilities: Calibrated probabilities of bankruptcy.

    Returns:
        The Brier score; lower is better.
    """
    return float(brier_score_loss(labels, probabilities))


def calibrator_path(stem: str, scorer: str, config: dict[str, Any]) -> Path:
    """Locate the saved calibrator of one file and scorer.

    Args:
        stem: File name without extension, e.g. 5year.
        scorer: One of SCORERS.
        config: Parsed project configuration, for the results folder.

    Returns:
        results/calibration/{stem}_{scorer}_platt.json under the repository.
    """
    results = REPO_ROOT / config["paths"]["results_dir"]
    return results / "calibration" / f"{stem}_{scorer}_platt.json"


def save_calibrator(
    calibrator: PlattCalibrator, stem: str, config: dict[str, Any] | None = None
) -> Path:
    """Write a calibrator to its JSON file, overwriting any earlier one.

    Args:
        calibrator: The fitted calibrator.
        stem: File name without extension, e.g. 5year.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        The file written.
    """
    if config is None:
        config = load_config()
    path = calibrator_path(stem, calibrator.scorer, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(calibrator), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_calibrator(
    stem: str, scorer: str, config: dict[str, Any] | None = None
) -> PlattCalibrator:
    """Read the saved calibrator of one file and scorer.

    Args:
        stem: File name without extension, e.g. 5year.
        scorer: One of SCORERS.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        The calibrator.

    Raises:
        ValueError: If the file holds a calibrator for another scorer.
    """
    if config is None:
        config = load_config()
    path = calibrator_path(stem, scorer, config)
    calibrator = PlattCalibrator(**json.loads(path.read_text(encoding="utf-8")))
    if calibrator.scorer != scorer:
        raise ValueError(f"{path.name} holds a calibrator for {calibrator.scorer}")
    return calibrator


def calibrate_all(
    stem: str, config: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Fit, save and summarise the calibrators of all five scorers on one file.

    Reads each scorer's saved out of fold predictions, so every scorer must
    have run first.

    Args:
        stem: File name without extension, e.g. 5year.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        For each scorer: its slope and intercept, and the cross fitted out of
        fold Brier score with the Brier score of always predicting the
        bankrupt share, for comparison. JSON ready.
    """
    if config is None:
        config = load_config()
    summary = {}
    for scorer in SCORERS:
        predictions = load_oof_predictions(stem, scorer, config)
        calibrator = fit_from_oof(predictions)
        save_calibrator(calibrator, stem, config)
        labels = predictions["label"]
        summary[scorer] = {
            "slope": calibrator.slope,
            "intercept": calibrator.intercept,
            "oof_brier": brier_score(labels, cross_fitted_probabilities(predictions)),
            "prevalence_brier": brier_score(
                labels, np.full(len(labels), labels.mean())
            ),
        }
    return summary


def reliability_diagram(
    curves: dict[str, tuple[np.ndarray | pd.Series, np.ndarray | pd.Series]],
    title: str,
) -> Figure:
    """Plot observed bankrupt share against mean predicted probability.

    Each scorer's rows are put into N_BINS quantile bins of predicted
    probability. A well calibrated scorer lies on the diagonal.

    Args:
        curves: For each scorer name, its labels and calibrated
            probabilities.
        title: Figure title.

    Returns:
        The figure, for the caller to save or show.
    """
    figure, axes = plt.subplots(figsize=(5, 5))
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect")
    for scorer, (labels, probabilities) in curves.items():
        observed, predicted = calibration_curve(
            labels, probabilities, n_bins=N_BINS, strategy="quantile"
        )
        axes.plot(predicted, observed, marker="o", label=scorer)
    axes.set_xlabel("Mean predicted probability")
    axes.set_ylabel("Observed bankrupt share")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_title(title)
    axes.legend()
    figure.tight_layout()
    return figure


def save_figure(
    figure: Figure, name: str, config: dict[str, Any] | None = None
) -> Path:
    """Save a figure as a 300 dpi PNG in results/figures.

    Args:
        figure: The figure.
        name: File name without extension.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        The file written.
    """
    if config is None:
        config = load_config()
    path = REPO_ROOT / config["paths"]["results_dir"] / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIGURE_DPI)
    return path
