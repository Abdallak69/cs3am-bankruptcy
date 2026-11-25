"""mlp_model.py builds and trains the PyTorch MLP, makes out of fold predictions
that pass the shared harness, fits the preprocessor on inner fitting rows only,
repeats exactly for a fixed seed and draws the random search from config.yaml.
A median imputer stands in for the project preprocessor, which is not written
yet, and the epoch limits are cut so the tests run in seconds."""

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from torch import nn

import src.mlp_model as mlp
from src.config import load_config
from src.evaluate import validate_oof_predictions
from src.load import FEATURES
from src.split import cv_folds, load_training_set

CONFIG = load_config()
STEM = "5year"

# Short runs: enough to exercise early stopping without a long test.
FAST = copy.deepcopy(CONFIG)
FAST["mlp"]["max_epochs"] = 6
FAST["mlp"]["patience"] = 2

PARAMS = mlp.MLPParams(
    class_weighting=True,
    n_layers=2,
    width=64,
    dropout=0.1,
    learning_rate=0.001,
    weight_decay=0.0001,
)


def median_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median").set_output(transform="pandas")


class RecordingImputer(SimpleImputer):
    """Median imputer that appends the row_ids of every fit to self.log."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RecordingImputer":
        self.log.append(X.index)
        return super().fit(X, y)


def small_problem(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(400, 5)).astype(np.float32)
    y = (x[:, 0] + 0.5 * rng.normal(size=400) > 1.0).astype(np.float32)
    return x, y


@pytest.fixture(scope="module")
def oof_run() -> tuple[pd.DataFrame, list[int]]:
    return mlp.oof_mlp(STEM, median_imputer, PARAMS, FAST)


def test_network_has_the_configured_shape() -> None:
    network = mlp.build_network(10, PARAMS)
    linear = [layer for layer in network if isinstance(layer, nn.Linear)]
    dropout = [layer for layer in network if isinstance(layer, nn.Dropout)]

    assert [(l.in_features, l.out_features) for l in linear] == [
        (10, 64),
        (64, 64),
        (64, 1),
    ]
    assert all(layer.p == 0.1 for layer in dropout) and len(dropout) == 2


def test_pos_weight_is_negatives_over_positives_or_one() -> None:
    labels = np.array([0, 0, 0, 1], dtype=np.float32)
    off = mlp.MLPParams(**{**PARAMS.__dict__, "class_weighting": False})

    assert mlp.pos_weight(labels, PARAMS).item() == pytest.approx(3.0)
    assert mlp.pos_weight(labels, off).item() == 1.0


def test_same_seed_gives_identical_scores() -> None:
    x, y = small_problem()
    first, _ = mlp.train_network(x, y, PARAMS, 42, FAST, n_epochs=3)
    second, _ = mlp.train_network(x, y, PARAMS, 42, FAST, n_epochs=3)
    other, _ = mlp.train_network(x, y, PARAMS, 43, FAST, n_epochs=3)

    assert np.array_equal(mlp.predict_logits(first, x), mlp.predict_logits(second, x))
    assert not np.array_equal(
        mlp.predict_logits(first, x), mlp.predict_logits(other, x)
    )


def test_early_stopping_returns_an_epoch_within_the_limit() -> None:
    x, y = small_problem()
    stop_x, stop_y = small_problem(seed=1)
    _, best_epoch = mlp.train_network(
        x, y, PARAMS, 42, FAST, stop_x=stop_x, stop_y=stop_y
    )

    assert 1 <= best_epoch <= FAST["mlp"]["max_epochs"]


def test_training_needs_stopping_rows_or_an_epoch_count() -> None:
    x, y = small_problem()
    with pytest.raises(ValueError):
        mlp.train_network(x, y, PARAMS, 42, FAST)


def test_predictions_pass_validation(oof_run: tuple[pd.DataFrame, list[int]]) -> None:
    predictions, best_epochs = oof_run

    assert (predictions["scorer"] == "mlp").all()
    validate_oof_predictions(predictions, STEM, CONFIG)
    assert len(best_epochs) == len(cv_folds(STEM, CONFIG))


def test_preprocessor_is_fitted_on_inner_fitting_rows_only() -> None:
    log: list[pd.Index] = []

    def recording_imputer() -> RecordingImputer:
        imputer = RecordingImputer(strategy="median").set_output(transform="pandas")
        imputer.log = log
        return imputer

    config = copy.deepcopy(FAST)
    config["mlp"]["max_epochs"] = 1
    mlp.oof_mlp(STEM, recording_imputer, PARAMS, config)

    folds = cv_folds(STEM, config)
    assert len(log) == len(folds)
    for fitted_on, (_, _, fit_row_ids, score_row_ids) in zip(log, folds):
        # Inside the fold's fitting rows, minus the 15% early stopping rows.
        assert fitted_on.isin(fit_row_ids).all()
        assert fitted_on.intersection(score_row_ids).empty
        expected = len(fit_row_ids) * (1 - config["cv"]["inner_val_fraction"])
        assert len(fitted_on) == pytest.approx(expected, abs=1)


def test_search_space_comes_from_config() -> None:
    configurations = mlp.sample_configurations(CONFIG)
    space = CONFIG["mlp"]["search"]
    low, high = space["weight_decay"]

    assert len(configurations) == CONFIG["mlp"]["n_configurations"]
    assert configurations == mlp.sample_configurations(CONFIG)
    for params in configurations:
        assert params.class_weighting in space["class_weighting"]
        assert params.n_layers in space["n_layers"]
        assert params.width in space["width"]
        assert params.dropout in space["dropout"]
        assert params.learning_rate in space["learning_rate"]
        assert low <= params.weight_decay <= high


def test_search_chooses_the_highest_mean_pr_auc() -> None:
    config = copy.deepcopy(FAST)
    config["mlp"]["max_epochs"] = 2
    configurations = mlp.sample_configurations(CONFIG)[:2]

    chosen, results, _, n_epochs = mlp.search_mlp(
        STEM, median_imputer, config, configurations
    )

    best = results["mean_pr_auc"].idxmax()
    assert chosen == configurations[best]
    assert n_epochs == results.loc[best, "median_best_epoch"]


def test_median_epoch_is_a_whole_epoch_of_at_least_one() -> None:
    assert mlp.median_epoch([10, 20, 31]) == 20
    assert mlp.median_epoch([10, 21]) == 16
    assert mlp.median_epoch([0, 0]) == 1


def test_final_models_cover_every_seed() -> None:
    training = load_training_set(STEM, CONFIG)
    models = mlp.fit_final_models(training, median_imputer, PARAMS, 2, FEATURES, CONFIG)

    assert sorted(models) == CONFIG["seeds"]["models"]
    scores = {seed: model.score(training[FEATURES]) for seed, model in models.items()}
    assert all(len(s) == len(training) for s in scores.values())
    assert not np.array_equal(scores[42], scores[43])
