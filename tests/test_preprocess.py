"""preprocess.py cleans the ratios with numbers learned on the fitting rows
only, keeps the five Altman inputs under their own names and does not
standardise."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import src.baselines as baselines
from src.config import load_config
from src.evaluate import validate_oof_predictions
from src.load import FEATURES
from src.preprocess import ALTMAN_INPUTS, make_preprocessor

CONFIG = load_config()
STEM = "5year"


def make_frame(rows: int = 200, seed: int = 0) -> pd.DataFrame:
    """Return 64 ratios of random normal values named Attr1 to Attr64."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(rows, 64)), columns=FEATURES)


def test_altman_inputs_match_baselines() -> None:
    assert list(ALTMAN_INPUTS) == baselines.ALTMAN_INPUTS


def test_returns_unfitted_dataframe_transformer_that_keeps_names() -> None:
    preprocessor = make_preprocessor()
    assert not hasattr(preprocessor, "kept_")

    frame = make_frame()
    out = clone(preprocessor).fit(frame).transform(frame)

    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(frame.index)
    assert set(out.columns) <= set(FEATURES)


def test_each_call_returns_a_new_preprocessor() -> None:
    assert make_preprocessor() is not make_preprocessor()


def test_infinite_and_missing_values_are_removed() -> None:
    frame = make_frame()
    frame.loc[0, "Attr4"] = np.inf
    frame.loc[1, "Attr5"] = -np.inf
    frame.loc[2, "Attr6"] = np.nan

    out = make_preprocessor().fit(frame).transform(frame)

    assert np.isfinite(out.to_numpy()).all()


def test_indicator_only_for_ratios_over_five_percent_missing() -> None:
    frame = make_frame()
    frame.loc[:19, "Attr20"] = np.nan  # 10% missing
    frame.loc[:5, "Attr21"] = np.nan  # 3% missing

    out = make_preprocessor().fit(frame).transform(frame)

    assert [c for c in out.columns if c.endswith("_missing")] == ["Attr20_missing"]
    assert out["Attr20_missing"].sum() == 20


def test_winsorises_at_the_training_percentiles() -> None:
    frame = make_frame(rows=1000)
    out = make_preprocessor().fit(frame).transform(frame)

    for name in ("Attr1", "Attr2"):
        assert out[name].min() == pytest.approx(frame[name].quantile(0.01))
        assert out[name].max() == pytest.approx(frame[name].quantile(0.99))


def test_does_not_standardise_the_altman_inputs() -> None:
    frame = make_frame()
    frame[list(ALTMAN_INPUTS)] = frame[list(ALTMAN_INPUTS)] * 10 + 100

    out = make_preprocessor().fit(frame).transform(frame)

    assert (out[list(ALTMAN_INPUTS)].mean() > 90).all()
    assert (out[list(ALTMAN_INPUTS)].std() > 5).all()


def test_learns_only_from_the_fitting_rows() -> None:
    fit_rows = make_frame(seed=1)
    fit_rows["Attr1"] = 5.0
    score_rows = make_frame(seed=2)
    score_rows["Attr1"] = np.nan  # would be filled with 0 if refitted

    out = make_preprocessor().fit(fit_rows).transform(score_rows)

    assert (out["Attr1"] == 5.0).all()


def test_altman_input_is_kept_and_its_partner_dropped() -> None:
    frame = make_frame()
    frame["Attr10"] = frame["Attr3"] * 2  # Spearman correlation of 1

    out = make_preprocessor().fit(frame).transform(frame)

    assert "Attr3" in out.columns
    assert "Attr10" not in out.columns


def test_two_altman_inputs_that_correlate_both_stay() -> None:
    frame = make_frame()
    frame["Attr7"] = frame["Attr6"] * 3

    out = make_preprocessor().fit(frame).transform(frame)

    assert {"Attr6", "Attr7"} <= set(out.columns)


def test_the_less_missing_ratio_of_a_pair_is_kept() -> None:
    frame = make_frame()
    frame["Attr12"] = frame["Attr11"]
    frame.loc[:9, "Attr11"] = np.nan  # more missing than Attr12

    out = make_preprocessor().fit(frame).transform(frame)

    assert "Attr12" in out.columns
    assert "Attr11" not in out.columns


def test_original_z_runs_through_baselines_with_this_preprocessor() -> None:
    predictions = baselines.oof_original_z(STEM, make_preprocessor, CONFIG)

    validate_oof_predictions(predictions, STEM, CONFIG)
