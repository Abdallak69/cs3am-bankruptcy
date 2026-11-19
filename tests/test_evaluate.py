"""evaluate.py checks out of fold predictions against the shared folds, saves
and loads them, and computes and writes their cross validation metrics."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.evaluate as evaluate
from src.config import REPO_ROOT, load_config
from src.load import LABEL
from src.split import cv_folds, load_training_set

CONFIG = load_config()
STEM = "5year"
SCORER = "logistic"


@pytest.fixture(scope="module")
def predictions() -> pd.DataFrame:
    # Seeded random scores in the prediction format, one part per fold.
    training = load_training_set(STEM, CONFIG)
    rng = np.random.default_rng(CONFIG["seeds"]["nominated"])
    parts = [
        pd.DataFrame(
            {
                "row_id": score_row_ids.to_numpy(),
                "label": training.loc[score_row_ids, LABEL].to_numpy(),
                "scorer": SCORER,
                "repeat": repeat,
                "fold": fold,
                "score": rng.random(len(score_row_ids)),
            }
        )
        for repeat, fold, _, score_row_ids in cv_folds(STEM, CONFIG)
    ]
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def scratch_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Writes go to an empty temporary root, never to the real results/.
    monkeypatch.setattr(evaluate, "REPO_ROOT", tmp_path)
    return tmp_path


def test_valid_predictions_pass(predictions: pd.DataFrame) -> None:
    evaluate.validate_oof_predictions(predictions, STEM, CONFIG)


def test_save_then_load_returns_identical_frame(
    predictions: pd.DataFrame, scratch_root: Path
) -> None:
    path = evaluate.save_oof_predictions(predictions, STEM, CONFIG)
    loaded = evaluate.load_oof_predictions(STEM, SCORER, CONFIG)

    assert path.is_relative_to(scratch_root)
    pd.testing.assert_frame_equal(loaded, predictions, check_exact=True)


def drop_fold_column(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns="fold")


def unknown_scorer(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(scorer="random_forest")


def two_scorers(frame: pd.DataFrame) -> pd.DataFrame:
    frame.loc[0, "scorer"] = "mlp"
    return frame


def infinite_score(frame: pd.DataFrame) -> pd.DataFrame:
    frame.loc[0, "score"] = np.inf
    return frame


def missing_row(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(index=0)


def row_in_wrong_fold(frame: pd.DataFrame) -> pd.DataFrame:
    frame.loc[0, "fold"] = 1
    return frame


def wrong_label(frame: pd.DataFrame) -> pd.DataFrame:
    frame.loc[0, "label"] = 1 - frame.loc[0, "label"]
    return frame


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (drop_fold_column, "expected columns"),
        (unknown_scorer, "unknown scorer"),
        (two_scorers, "one scorer per frame"),
        (infinite_score, "must be finite"),
        (missing_row, "repeat 0 fold 0 must hold exactly"),
        (row_in_wrong_fold, "repeat 0 fold 0 must hold exactly"),
        (wrong_label, "label that differs"),
    ],
)
def test_validation_rejects(
    predictions: pd.DataFrame,
    corrupt: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate.validate_oof_predictions(corrupt(predictions.copy()), STEM, CONFIG)


def test_cv_metrics_gives_fifteen_folds_between_zero_and_one(
    predictions: pd.DataFrame,
) -> None:
    metrics = evaluate.cv_metrics(predictions)

    assert metrics["scorer"] == SCORER
    assert len(metrics["folds"]) == 15
    for entry in metrics["folds"]:
        assert 0 <= entry["pr_auc"] <= 1
        assert 0 <= entry["roc_auc"] <= 1
        assert entry["pr_auc_over_share"] == entry["pr_auc"] / entry["bankrupt_share"]
    assert set(metrics["mean"]) == set(metrics["std"]) == set(evaluate.FOLD_METRICS)


def test_perfect_score_gives_auc_of_one(predictions: pd.DataFrame) -> None:
    perfect = predictions.assign(score=predictions["label"].astype(float))
    metrics = evaluate.cv_metrics(perfect)

    for entry in metrics["folds"]:
        assert entry["pr_auc"] == entry["roc_auc"] == 1
    assert metrics["mean"]["pr_auc"] == metrics["mean"]["roc_auc"] == 1


def test_write_metrics_json(predictions: pd.DataFrame, scratch_root: Path) -> None:
    metrics = evaluate.cv_metrics(predictions)
    name = "test_write_metrics_json"
    path = evaluate.write_metrics_json(metrics, name, CONFIG)

    assert path == scratch_root / CONFIG["paths"]["metrics_dir"] / f"{name}.json"
    assert json.loads(path.read_text(encoding="utf-8")) == metrics
    assert not (REPO_ROOT / CONFIG["paths"]["metrics_dir"] / f"{name}.json").exists()
