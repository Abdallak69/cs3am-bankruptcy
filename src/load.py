"""Load the five UCI Polish bankruptcy ARFF files.

Reads each file from the raw data directory, decodes the class label
(1 = bankrupt), checks the file against the published UCI counts, and removes
exact duplicate rows from each file before any split, logging the number
removed. The five files are never pooled.

Owner: Abdalla (project base).
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.io import arff

from src.config import REPO_ROOT, load_config

logger = logging.getLogger(__name__)

FEATURES: list[str] = [f"Attr{number}" for number in range(1, 65)]
LABEL: str = "class"


def read_arff(path: Path) -> pd.DataFrame:
    """Read one ARFF file into a DataFrame.

    Args:
        path: Location of the ARFF file.

    Returns:
        One row per statement, indexed by its position in the file as
        row_id. The ratios are floats, with ARFF "?" read as NaN and infinite
        values left as they are. The class column is an int, 1 = bankrupt.
    """
    data, _ = arff.loadarff(path)
    frame = pd.DataFrame(data)
    # loadarff returns the nominal class column as bytes, e.g. b"1".
    frame[LABEL] = frame[LABEL].str.decode("utf-8").astype(int)
    frame.index.name = "row_id"
    return frame


def check_raw_file(frame: pd.DataFrame, stem: str, expected: dict[str, int]) -> None:
    """Check a raw file against its published UCI counts.

    Args:
        frame: The file as returned by read_arff, before duplicate removal.
        stem: File name without extension, e.g. 5year, used in messages.
        expected: Published counts with the keys rows and bankrupt.

    Raises:
        ValueError: If the columns are not Attr1 to Attr64 followed by class,
            a label is not 0 or 1, or the row or bankrupt count differs from
            the published figure.
    """
    if list(frame.columns) != FEATURES + [LABEL]:
        raise ValueError(f"{stem}: expected columns Attr1 to Attr64 then class")

    unexpected = set(frame[LABEL].unique()) - {0, 1}
    if unexpected:
        raise ValueError(f"{stem}: labels must be 0 or 1, found {unexpected}")

    rows = len(frame)
    bankrupt = int(frame[LABEL].sum())
    if rows != expected["rows"] or bankrupt != expected["bankrupt"]:
        raise ValueError(
            f"{stem}: expected {expected['rows']} rows and {expected['bankrupt']} "
            f"bankrupt as published by UCI, found {rows} rows and {bankrupt} "
            "bankrupt"
        )


def remove_duplicates(frame: pd.DataFrame, stem: str) -> pd.DataFrame:
    """Drop exact duplicate rows, keeping the first, and log what was found.

    Two rows are exact duplicates when all 64 ratios and the label match;
    pandas treats NaN in the same column as a match. Rows whose ratios match
    but whose labels differ are kept and only counted.

    Args:
        frame: One raw file as returned by read_arff.
        stem: File name without extension, used in log messages.

    Returns:
        The file without exact duplicates, keeping the original row_id.
    """
    duplicated = frame.duplicated(keep="first")
    deduplicated = frame[~duplicated]
    # Once exact duplicates are gone, rows that still share all 64 ratios
    # must have different labels.
    conflicting = deduplicated.duplicated(subset=FEATURES, keep=False)

    logger.info("%s: removed %d exact duplicate rows", stem, duplicated.sum())
    logger.info(
        "%s: %d rows share all 64 ratios with a row of the other label (kept)",
        stem,
        conflicting.sum(),
    )
    return deduplicated


def load_file(stem: str, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load, check and deduplicate one raw ARFF file.

    Args:
        stem: File name without extension, e.g. 5year.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        The file without exact duplicates, as described in read_arff.

    Raises:
        ValueError: If the raw file does not match its published counts.
    """
    if config is None:
        config = load_config()
    path = REPO_ROOT / config["paths"]["raw_data_dir"] / f"{stem}.arff"

    frame = read_arff(path)
    check_raw_file(frame, stem, config["raw_counts"][stem])
    frame = remove_duplicates(frame, stem)

    bankrupt = int(frame[LABEL].sum())
    logger.info(
        "%s: %d rows and %d bankrupt after duplicate removal, class weight %.2f",
        stem,
        len(frame),
        bankrupt,
        (len(frame) - bankrupt) / bankrupt,
    )
    return frame


def load_all(config: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Load each of the five raw files separately.

    The files are returned side by side and never pooled or concatenated.

    Args:
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        One DataFrame per file, keyed by file stem (1year to 5year).
    """
    if config is None:
        config = load_config()
    return {stem: load_file(stem, config) for stem in config["raw_counts"]}
