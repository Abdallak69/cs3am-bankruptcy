"""calibrate.py fits Platt calibrators on out of fold predictions, saves and
reloads them, cross fits each fold on the other folds of its repeat and draws
reliability diagrams. Out of fold predictions of the original Z' and the
logistic regression, with a median imputer as preprocessor, stand in for the
scorers that are not written yet. Files go to a temporary folder."""

import copy
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

matplotlib.use("Agg")

import src.baselines as baselines  # noqa: E402
import src.calibrate as calibrate  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluate import SCORERS, save_oof_predictions  # noqa: E402

CONFIG = load_config()
STEM = "5year"


def median_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median").set_output(transform="pandas")


@pytest.fixture(scope="module")
def oof() -> dict[str, pd.DataFrame]:
    return {
        "original_z": baselines.oof_original_z(STEM, median_imputer, CONFIG),
        "logistic": baselines.oof_logistic_at_c(
            STEM, "logistic", median_imputer, 1.0, CONFIG
        ),
    }


@pytest.fixture
def tmp_config(tmp_path: Path) -> dict[str, Any]:
    config = copy.deepcopy(CONFIG)
    config["paths"]["results_dir"] = str(tmp_path)
    config["paths"]["oof_predictions_pattern"] = str(
        tmp_path / "oof" / "{stem}_{scorer}.csv"
    )
    return config


def test_fit_recovers_known_slope_and_intercept() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=20_000)
    labels = rng.random(20_000) < 1 / (1 + np.exp(-(2.0 * scores - 3.0)))

    calibrator = calibrate.fit_platt(scores, labels, "mlp")

    assert calibrator.slope == pytest.approx(2.0, abs=0.1)
    assert calibrator.intercept == pytest.approx(-3.0, abs=0.1)


def test_predict_is_the_logistic_function() -> None:
    calibrator = calibrate.PlattCalibrator("mlp", slope=2.0, intercept=-1.0)

    assert calibrator.predict([0.5, 0.0]).tolist() == pytest.approx(
        [0.5, 1 / (1 + np.exp(1.0))]
    )


@pytest.mark.parametrize("scorer", ["original_z", "logistic"])
def test_calibration_keeps_the_ranking_and_matches_the_bankrupt_share(
    oof: dict[str, pd.DataFrame], scorer: str
) -> None:
    predictions = oof[scorer]
    calibrator = calibrate.fit_from_oof(predictions)
    probabilities = calibrator.predict(predictions["score"])

    # A higher score never gets a lower probability.
    assert calibrator.slope > 0
    # Logistic regression with an intercept matches the mean of the labels.
    assert probabilities.mean() == pytest.approx(predictions["label"].mean(), abs=1e-4)


def test_cross_fitting_never_uses_the_fold_it_calibrates(
    oof: dict[str, pd.DataFrame],
) -> None:
    predictions = oof["logistic"].copy()
    # Flip every label in repeat 0 fold 0. A calibrator that saw them would
    # change; one fitted on the other folds gives that fold the same output.
    flipped = predictions.copy()
    target = (flipped["repeat"] == 0) & (flipped["fold"] == 0)
    flipped.loc[target, "label"] = 1 - flipped.loc[target, "label"]

    original = calibrate.cross_fitted_probabilities(predictions)
    changed = calibrate.cross_fitted_probabilities(flipped)

    assert np.allclose(original[target], changed[target])
    assert original.notna().all()


def test_saved_calibrator_reloads_unchanged(tmp_config: dict[str, Any]) -> None:
    calibrator = calibrate.PlattCalibrator("xgboost", slope=0.7, intercept=-2.5)

    path = calibrate.save_calibrator(calibrator, STEM, tmp_config)

    assert path.name == "5year_xgboost_platt.json"
    assert calibrate.load_calibrator(STEM, "xgboost", tmp_config) == calibrator


def test_calibrate_all_covers_every_scorer(
    oof: dict[str, pd.DataFrame], tmp_config: dict[str, Any]
) -> None:
    # Reuse two real runs under all five names, since three scorers are not
    # written yet.
    for scorer in SCORERS:
        source = oof["original_z"] if scorer == "original_z" else oof["logistic"]
        save_oof_predictions(source.assign(scorer=scorer), STEM, tmp_config)

    summary = calibrate.calibrate_all(STEM, tmp_config)

    assert sorted(summary) == sorted(SCORERS)
    for scorer, entry in summary.items():
        assert calibrate.calibrator_path(STEM, scorer, tmp_config).is_file()
        # The logistic regression ranks well enough to beat the base rate.
        if scorer != "original_z":
            assert entry["oof_brier"] < entry["prevalence_brier"]


def test_reliability_diagram_saves_a_png(
    oof: dict[str, pd.DataFrame], tmp_config: dict[str, Any]
) -> None:
    predictions = oof["logistic"]
    probabilities = calibrate.cross_fitted_probabilities(predictions)
    figure = calibrate.reliability_diagram(
        {"logistic": (predictions["label"], probabilities)}, "test"
    )

    path = calibrate.save_figure(figure, "reliability_test", tmp_config)

    assert path.is_file() and path.suffix == ".png"
