"""Raw row and bankrupt counts of the five ARFF files, as published by UCI.

These are counts before duplicate removal. If one fails, the raw file is not
the published one; do not edit the expected numbers.
"""

import pytest
from scipy.io import arff

from src.config import REPO_ROOT, load_config

EXPECTED_COUNTS = [
    ("1year", 7027, 271),
    ("2year", 10173, 400),
    ("3year", 10503, 495),
    ("4year", 9792, 515),
    ("5year", 5910, 410),
]


@pytest.mark.parametrize("stem, expected_rows, expected_bankrupt", EXPECTED_COUNTS)
def test_raw_counts(stem: str, expected_rows: int, expected_bankrupt: int) -> None:
    raw_dir = REPO_ROOT / load_config()["paths"]["raw_data_dir"]
    data, _ = arff.loadarff(raw_dir / f"{stem}.arff")

    # loadarff returns the nominal class column as bytes.
    assert set(data["class"]) == {b"0", b"1"}
    assert len(data) == expected_rows
    assert int((data["class"] == b"1").sum()) == expected_bankrupt
