"""load.py returns the five files separately, checked and without exact
duplicates, with row_id pointing at the original statement."""

import pandas as pd
import pytest

from src.config import REPO_ROOT, load_config
from src.load import FEATURES, LABEL, check_raw_file, load_all, read_arff

STEMS = ["1year", "2year", "3year", "4year", "5year"]
RAW_COUNTS = load_config()["raw_counts"]


def read_raw(stem: str) -> pd.DataFrame:
    raw_dir = REPO_ROOT / load_config()["paths"]["raw_data_dir"]
    return read_arff(raw_dir / f"{stem}.arff")


@pytest.fixture(scope="module")
def files() -> dict[str, pd.DataFrame]:
    return load_all()


def test_five_separate_files(files: dict[str, pd.DataFrame]) -> None:
    assert list(files) == STEMS
    # A pooled frame would hold more rows than any single raw file.
    for stem, frame in files.items():
        assert len(frame) <= RAW_COUNTS[stem]["rows"]


@pytest.mark.parametrize("stem", STEMS)
def test_no_exact_duplicates(files: dict[str, pd.DataFrame], stem: str) -> None:
    assert not files[stem].duplicated().any()


@pytest.mark.parametrize("stem", STEMS)
def test_row_id_unique_and_in_range(files: dict[str, pd.DataFrame], stem: str) -> None:
    row_id = files[stem].index
    assert row_id.name == "row_id"
    assert row_id.is_unique
    assert row_id.min() >= 0
    assert row_id.max() < RAW_COUNTS[stem]["rows"]


def test_row_id_points_at_raw_statement(files: dict[str, pd.DataFrame]) -> None:
    loaded = files["5year"]
    pd.testing.assert_frame_equal(loaded, read_raw("5year").loc[loaded.index])


@pytest.mark.parametrize("stem", STEMS)
def test_64_float_features(files: dict[str, pd.DataFrame], stem: str) -> None:
    frame = files[stem]
    assert list(frame.columns) == FEATURES + [LABEL]
    assert len(FEATURES) == 64
    assert all(pd.api.types.is_float_dtype(frame[column]) for column in FEATURES)


@pytest.mark.parametrize("stem", STEMS)
def test_label_is_int_0_or_1(files: dict[str, pd.DataFrame], stem: str) -> None:
    label = files[stem][LABEL]
    assert pd.api.types.is_integer_dtype(label)
    assert set(label) == {0, 1}


def test_count_check_raises_on_wrong_count() -> None:
    wrong = dict(RAW_COUNTS["5year"], rows=RAW_COUNTS["5year"]["rows"] + 1)
    with pytest.raises(ValueError, match="5year: expected 5911 rows"):
        check_raw_file(read_raw("5year"), "5year", wrong)
