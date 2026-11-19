"""Out of fold predictions of the three baselines on the shared folds.

Original Z' applies Altman's published coefficients to the preprocessed
ratios, with -Z' as the risk score; nothing but the preprocessor is fitted.
Refitted Z' is a logistic regression on the five Altman inputs only, and the
statistical baseline is a logistic regression on every column the
preprocessor outputs. Both logistic regressions standardise their inputs and
use an L2 penalty and balanced class weights, with C chosen from
baselines.c_grid in config.yaml by mean out of fold PR AUC.

This module holds no preprocessing of its own. Every function takes a
preprocessor: a callable with no arguments that returns an unfitted sklearn
transformer, which takes and returns a DataFrame with its column names, keeps
the five Altman inputs and does not standardise. In each fold a new one is
fitted on the fitting rows only and applied to the scored rows.

Nothing is saved here; save_oof_predictions in evaluate.py does that.

Owner: Abdalla.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import load_config
from src.evaluate import PREDICTION_COLUMNS, cv_metrics
from src.load import FEATURES, LABEL
from src.split import cv_folds, load_training_set

logger = logging.getLogger(__name__)

Preprocessor = Callable[[], TransformerMixin]

# Altman, E. I. (1983), Corporate Financial Distress, Wiley: the Z' model for
# private firms. X1 to X5 map to the UCI ratios as Attr3 working capital /
# total assets, Attr6 retained earnings / total assets, Attr7 EBIT / total
# assets, Attr8 book equity / total liabilities, Attr9 sales / total assets.
ALTMAN_INPUTS: list[str] = ["Attr3", "Attr6", "Attr7", "Attr8", "Attr9"]
Z_PRIME_COEFFICIENTS: list[float] = [0.717, 0.847, 3.107, 0.420, 0.998]

LOGISTIC_SCORERS: tuple[str, ...] = ("refitted_z", "logistic")
# lbfgs stops earlier once it converges; this only caps the iterations.
MAX_ITER: int = 1_000


def original_z_scores(frame: pd.DataFrame) -> pd.Series:
    """Compute the original Z' risk score of each row.

    Args:
        frame: Rows holding the five ALTMAN_INPUTS columns, after
            preprocessing.

    Returns:
        -Z' for each row, indexed like frame, so a higher score means more
        likely bankrupt. A row missing an Altman input gets NaN.
    """
    return -(frame[ALTMAN_INPUTS] @ np.array(Z_PRIME_COEFFICIENTS))


def select_altman_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the five Altman inputs, for the refitted Z'.

    Args:
        frame: Preprocessed rows with their column names.

    Returns:
        The ALTMAN_INPUTS columns of frame, in that order.
    """
    return frame[ALTMAN_INPUTS]


def logistic_pipeline(
    scorer: str, preprocessor: Preprocessor, c: float, config: dict[str, Any]
) -> Pipeline:
    """Build the unfitted pipeline of one logistic regression baseline.

    Args:
        scorer: refitted_z to use the five Altman inputs only, or logistic to
            use every column the preprocessor outputs.
        preprocessor: Returns a new unfitted preprocessing transformer.
        c: Inverse regularisation strength of the L2 penalty.
        config: Parsed project configuration, for the nominated seed.

    Returns:
        A Pipeline with the steps preprocess, select, scale and model.

    Raises:
        ValueError: If scorer is not in LOGISTIC_SCORERS.
    """
    if scorer not in LOGISTIC_SCORERS:
        raise ValueError(f"expected one of {LOGISTIC_SCORERS}, found {scorer!r}")
    select = (
        FunctionTransformer(select_altman_inputs)
        if scorer == "refitted_z"
        else "passthrough"
    )
    model = LogisticRegression(
        C=c,
        l1_ratio=0.0,  # pure L2; scikit-learn 1.8 deprecated penalty="l2"
        class_weight="balanced",
        random_state=config["seeds"]["nominated"],
        max_iter=MAX_ITER,
    )
    return Pipeline(
        [
            ("preprocess", preprocessor()),
            ("select", select),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def oof_scores(
    stem: str,
    scorer: str,
    fit_and_score: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Score the rows of every fold from cv_folds with a model fitted on the rest.

    Args:
        stem: File name without extension, e.g. 5year.
        scorer: Scorer name written to the scorer column.
        fit_and_score: Takes the fitting rows' ratios, their labels and the
            scored rows' ratios, fits on the first two and returns one risk
            score per scored row.
        config: Parsed project configuration.

    Returns:
        The out of fold predictions in PREDICTION_COLUMNS.
    """
    training = load_training_set(stem, config)
    parts = []
    for repeat, fold, fit_row_ids, score_row_ids in cv_folds(stem, config):
        scores = fit_and_score(
            training.loc[fit_row_ids, FEATURES],
            training.loc[fit_row_ids, LABEL],
            training.loc[score_row_ids, FEATURES],
        )
        parts.append(
            pd.DataFrame(
                {
                    "row_id": score_row_ids.to_numpy(),
                    "label": training.loc[score_row_ids, LABEL].to_numpy(),
                    "scorer": scorer,
                    "repeat": repeat,
                    "fold": fold,
                    "score": np.asarray(scores, dtype=float),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)[PREDICTION_COLUMNS]


def oof_original_z(
    stem: str, preprocessor: Preprocessor, config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Make the out of fold predictions of the original Z'.

    Args:
        stem: File name without extension, e.g. 5year.
        preprocessor: Returns a new unfitted preprocessing transformer.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        The out of fold predictions in PREDICTION_COLUMNS, scorer original_z.
    """
    if config is None:
        config = load_config()

    def fit_and_score(
        fit_x: pd.DataFrame, fit_y: pd.Series, score_x: pd.DataFrame
    ) -> pd.Series:
        transformer = preprocessor().fit(fit_x, fit_y)
        return original_z_scores(transformer.transform(score_x))

    return oof_scores(stem, "original_z", fit_and_score, config)


def oof_logistic_at_c(
    stem: str,
    scorer: str,
    preprocessor: Preprocessor,
    c: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Make the out of fold predictions of one logistic regression at one C.

    Args:
        stem: File name without extension, e.g. 5year.
        scorer: refitted_z or logistic, as in logistic_pipeline.
        preprocessor: Returns a new unfitted preprocessing transformer.
        c: Inverse regularisation strength of the L2 penalty.
        config: Parsed project configuration.

    Returns:
        The out of fold predictions in PREDICTION_COLUMNS, scored with the
        decision function (the log odds of bankruptcy).
    """

    def fit_and_score(
        fit_x: pd.DataFrame, fit_y: pd.Series, score_x: pd.DataFrame
    ) -> np.ndarray:
        pipeline = logistic_pipeline(scorer, preprocessor, c, config)
        return pipeline.fit(fit_x, fit_y).decision_function(score_x)

    return oof_scores(stem, scorer, fit_and_score, config)


def oof_logistic(
    stem: str,
    scorer: str,
    preprocessor: Preprocessor,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Tune C for one logistic regression baseline and return its predictions.

    Every C in baselines.c_grid is run on all 15 folds, and the C with the
    highest mean PR AUC over the folds (from cv_metrics) is chosen. On a tie
    the smaller C is chosen.

    Args:
        stem: File name without extension, e.g. 5year.
        scorer: refitted_z or logistic, as in logistic_pipeline.
        preprocessor: Returns a new unfitted preprocessing transformer.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        The out of fold predictions of the chosen C in PREDICTION_COLUMNS,
        and the mean PR AUC of every C, indexed by C in ascending order.
    """
    if config is None:
        config = load_config()

    predictions = {}
    mean_pr_auc = {}
    for c in config["baselines"]["c_grid"]:
        predictions[c] = oof_logistic_at_c(stem, scorer, preprocessor, c, config)
        mean_pr_auc[c] = cv_metrics(predictions[c])["mean"]["pr_auc"]

    mean_pr_auc_by_c = pd.Series(mean_pr_auc, name="mean_pr_auc").sort_index()
    mean_pr_auc_by_c.index.name = "c"
    # idxmax returns the first highest value, which is the smaller C on a tie.
    best_c = mean_pr_auc_by_c.idxmax()
    logger.info(
        "%s %s: chose C = %s, mean PR AUC %.4f",
        stem,
        scorer,
        best_c,
        mean_pr_auc_by_c[best_c],
    )
    return predictions[best_c], mean_pr_auc_by_c
