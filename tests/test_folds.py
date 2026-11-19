"""cv_folds gives every scorer the same repeated stratified folds of the
training rows, and inner_split holds out early stopping rows from the fitting
rows of a fold only."""

import pandas as pd
import pytest

from src.config import load_config
from src.load import LABEL
from src.split import cv_folds, inner_split, load_training_set

CONFIG = load_config()
STEM = "5year"

Fold = tuple[int, int, pd.Index, pd.Index]


@pytest.fixture(scope="module")
def training() -> pd.DataFrame:
    return load_training_set(STEM, CONFIG)


@pytest.fixture(scope="module")
def folds() -> list[Fold]:
    return cv_folds(STEM, CONFIG)


def test_fifteen_folds_numbered(folds: list[Fold]) -> None:
    numbers = [(repeat, fold) for repeat, fold, _, _ in folds]
    assert numbers == [(repeat, fold) for repeat in range(3) for fold in range(5)]


def test_scored_rows_partition_each_repeat(
    training: pd.DataFrame, folds: list[Fold]
) -> None:
    for repeat in range(3):
        scored = [score for number, _, _, score in folds if number == repeat]
        combined = scored[0].append(scored[1:])
        assert combined.is_unique
        assert combined.sort_values().equals(training.index)


def test_fit_and_scored_rows_partition_training(
    training: pd.DataFrame, folds: list[Fold]
) -> None:
    for _, _, fit, score in folds:
        assert fit.name == score.name == "row_id"
        assert fit.is_monotonic_increasing and score.is_monotonic_increasing
        assert fit.intersection(score).empty
        assert fit.union(score).equals(training.index)


def test_scored_bankrupt_share_within_one_row(
    training: pd.DataFrame, folds: list[Fold]
) -> None:
    share = training[LABEL].mean()
    for _, _, _, score in folds:
        bankrupt = training.loc[score, LABEL].sum()
        assert abs(bankrupt - share * len(score)) <= 1


def test_repeats_give_different_partitions(folds: list[Fold]) -> None:
    partitions = {
        frozenset(frozenset(score) for number, _, _, score in folds if number == repeat)
        for repeat in range(3)
    }
    assert len(partitions) == 3


def test_cv_folds_repeatable(folds: list[Fold]) -> None:
    again = cv_folds(STEM, CONFIG)
    assert len(again) == len(folds)
    for first, second in zip(folds, again):
        assert first[:2] == second[:2]
        assert first[2].equals(second[2])
        assert first[3].equals(second[3])


def test_inner_split(training: pd.DataFrame, folds: list[Fold]) -> None:
    fraction = CONFIG["cv"]["inner_val_fraction"]
    for _, _, fit, score in folds:
        fit_rows = training.loc[fit]
        inner_fit, stop = inner_split(fit_rows, CONFIG)
        share = fit_rows[LABEL].mean()

        assert inner_fit.intersection(stop).empty
        assert inner_fit.union(stop).equals(fit)
        assert abs(len(stop) - fraction * len(fit)) <= 1
        assert abs(fit_rows.loc[stop, LABEL].sum() - share * len(stop)) <= 1
        assert stop.intersection(score).empty
