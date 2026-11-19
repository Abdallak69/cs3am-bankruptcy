"""baselines.py makes out of fold predictions of the original Z', the refitted
Z' and the logistic regression, fitting the preprocessor on fitting rows only
and choosing C by mean PR AUC. A median imputer stands in for the project
preprocessor, which is not written yet."""

import copy
from typing import Any

import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

import src.baselines as baselines
from src.config import load_config
from src.evaluate import cv_metrics, validate_oof_predictions
from src.load import FEATURES, LABEL
from src.split import cv_folds, load_training_set

CONFIG = load_config()
STEM = "5year"

# A deprecated argument or an unconverged fit fails the test.
pytestmark = pytest.mark.filterwarnings(
    "error::FutureWarning",
    "error::DeprecationWarning",
    "error::sklearn.exceptions.ConvergenceWarning",
)


def median_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median").set_output(transform="pandas")


class RecordingImputer(SimpleImputer):
    """Median imputer that appends the row_ids of every fit to self.log."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RecordingImputer":
        self.log.append(X.index)
        return super().fit(X, y)


@pytest.fixture(scope="module")
def runs() -> dict[str, tuple[pd.DataFrame, pd.Series | None]]:
    return {
        "original_z": (baselines.oof_original_z(STEM, median_imputer, CONFIG), None),
        "refitted_z": baselines.oof_logistic(
            STEM, "refitted_z", median_imputer, CONFIG
        ),
        "logistic": baselines.oof_logistic(STEM, "logistic", median_imputer, CONFIG),
    }


def test_original_z_scores_match_hand_computation() -> None:
    frame = pd.DataFrame(
        {
            "Attr1": [99.0, 99.0],  # not an Altman input, so ignored
            "Attr3": [0.1, 1.0],
            "Attr6": [0.2, 1.0],
            "Attr7": [0.3, 1.0],
            "Attr8": [0.4, 1.0],
            "Attr9": [0.5, 1.0],
        },
        index=pd.Index([7, 3], name="row_id"),
    )
    # Z' = 0.717*0.1 + 0.847*0.2 + 3.107*0.3 + 0.420*0.4 + 0.998*0.5 = 1.8402
    # Z' = 0.717 + 0.847 + 3.107 + 0.420 + 0.998 = 6.089
    scores = baselines.original_z_scores(frame)

    assert scores.index.equals(frame.index)
    assert scores.tolist() == pytest.approx([-1.8402, -6.089])


@pytest.mark.parametrize("scorer", ["original_z", "refitted_z", "logistic"])
def test_predictions_pass_validation(
    runs: dict[str, tuple[pd.DataFrame, pd.Series | None]], scorer: str
) -> None:
    predictions, _ = runs[scorer]

    assert (predictions["scorer"] == scorer).all()
    validate_oof_predictions(predictions, STEM, CONFIG)


def test_preprocessor_is_fitted_on_fitting_rows_only() -> None:
    log: list[pd.Index] = []

    def recording_imputer() -> RecordingImputer:
        imputer = RecordingImputer(strategy="median").set_output(transform="pandas")
        imputer.log = log
        return imputer

    config = copy.deepcopy(CONFIG)
    config["baselines"]["c_grid"] = [1.0]
    baselines.oof_original_z(STEM, recording_imputer, config)
    baselines.oof_logistic(STEM, "refitted_z", recording_imputer, config)
    baselines.oof_logistic(STEM, "logistic", recording_imputer, config)

    folds = cv_folds(STEM, config)
    # One fit per fold for each of the three runs, in fold order.
    assert len(log) == 3 * len(folds)
    for fitted_on, (_, _, fit_row_ids, score_row_ids) in zip(log, folds * 3):
        assert fitted_on.sort_values().equals(fit_row_ids)
        assert fitted_on.intersection(score_row_ids).empty


def test_refitted_z_uses_exactly_the_altman_inputs() -> None:
    training = load_training_set(STEM, CONFIG)
    _, _, fit_row_ids, _ = cv_folds(STEM, CONFIG)[0]
    fit_x, fit_y = training.loc[fit_row_ids, FEATURES], training.loc[fit_row_ids, LABEL]

    refitted = baselines.logistic_pipeline("refitted_z", median_imputer, 1.0, CONFIG)
    refitted.fit(fit_x, fit_y)
    full = baselines.logistic_pipeline("logistic", median_imputer, 1.0, CONFIG)
    full.fit(fit_x, fit_y)

    assert list(refitted["scale"].feature_names_in_) == baselines.ALTMAN_INPUTS
    assert refitted["model"].coef_.shape == (1, len(baselines.ALTMAN_INPUTS))
    assert full["model"].coef_.shape == (1, len(FEATURES))


@pytest.mark.parametrize("scorer", ["refitted_z", "logistic"])
def test_chosen_c_has_highest_mean_pr_auc(
    runs: dict[str, tuple[pd.DataFrame, pd.Series | None]], scorer: str
) -> None:
    predictions, mean_pr_auc_by_c = runs[scorer]

    assert mean_pr_auc_by_c.index.tolist() == sorted(CONFIG["baselines"]["c_grid"])
    assert cv_metrics(predictions)["mean"]["pr_auc"] == mean_pr_auc_by_c.max()


def test_c_grid_comes_from_config() -> None:
    config = copy.deepcopy(CONFIG)
    config["baselines"]["c_grid"] = [2.0, 0.5]

    _, mean_pr_auc_by_c = baselines.oof_logistic(
        STEM, "refitted_z", median_imputer, config
    )

    assert mean_pr_auc_by_c.index.tolist() == [0.5, 2.0]
