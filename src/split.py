"""Frozen stratified 80/20 train/test split and the shared cross validation folds.

Creates one stratified split per file with the nominated seed (42), after
duplicate removal, and writes the training and test row_ids once to
data/splits/. The 5year.arff split is reused by every experiment. Apart from
evaluate.py, this is the only module that touches the test indices, and it
only writes them. Every other module gets its data from load_training_set,
which reads the training indices only.

cv_folds gives every scorer the same repeated stratified folds of the
training rows, so their out of fold predictions are paired, and inner_split
holds out early stopping rows from the fitting rows of one fold. Neither
touches the test indices.

Owner: Abdalla (changes need Abdalla; Cameron as authorised deputy).
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from src.config import REPO_ROOT, load_config
from src.load import LABEL, load_file

logger = logging.getLogger(__name__)

ROW_ID: str = "row_id"


def split_file(
    frame: pd.DataFrame, stem: str, config: dict[str, Any]
) -> tuple[pd.Index, pd.Index]:
    """Make the stratified train/test split of one deduplicated file.

    This rebuilds the test row_ids, so only split.py and evaluate.py may call
    it; the test indices guard fails if any other module or a notebook
    refers to it. Everything else uses load_training_set.

    Args:
        frame: One file as returned by load_file.
        stem: File name without extension, e.g. 5year, used in log messages.
        config: Parsed project configuration, for the test size and the
            nominated seed.

    Returns:
        The training row_ids and the test row_ids, each sorted.
    """
    train, test = train_test_split(
        frame.index.to_numpy(),
        test_size=config["split"]["test_size"],
        random_state=config["seeds"]["nominated"],
        stratify=frame[LABEL],
    )
    train = pd.Index(train, name=ROW_ID).sort_values()
    test = pd.Index(test, name=ROW_ID).sort_values()

    for part, row_ids in (("training", train), ("test", test)):
        logger.info(
            "%s: %s set has %d rows and %d bankrupt",
            stem,
            part,
            len(row_ids),
            int(frame.loc[row_ids, LABEL].sum()),
        )
    return train, test


def train_indices_path(stem: str, config: dict[str, Any]) -> Path:
    """Locate the training indices file of one file.

    Args:
        stem: File name without extension, e.g. 5year.
        config: Parsed project configuration.

    Returns:
        The path of the CSV holding the training row_ids.
    """
    return REPO_ROOT / config["paths"]["train_indices_pattern"].format(stem=stem)


def write_splits(config: dict[str, Any] | None = None) -> list[Path]:
    """Split each of the five files and write its row_ids, once.

    Writes one CSV of training row_ids and one of test row_ids per file, each
    with a single sorted row_id column. Nothing is written unless this is
    called. Every split is made and every path checked before the first file
    is written, so a refused run, or one that fails before writing, leaves
    data/splits/ as it was.

    Args:
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        The ten files written.

    Raises:
        FileExistsError: If any of the ten files already exists. The split
            is frozen and is never overwritten.
    """
    if config is None:
        config = load_config()

    targets = {
        stem: (
            train_indices_path(stem, config),
            REPO_ROOT / config["paths"]["test_indices_pattern"].format(stem=stem),
        )
        for stem in config["raw_counts"]
    }
    existing = [
        str(path.relative_to(REPO_ROOT))
        for paths in targets.values()
        for path in paths
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "The split is frozen and written once; refusing to overwrite "
            + ", ".join(existing)
        )

    splits = {
        stem: split_file(load_file(stem, config), stem, config) for stem in targets
    }

    written = []
    for stem, paths in targets.items():
        for path, row_ids in zip(paths, splits[stem]):
            path.parent.mkdir(parents=True, exist_ok=True)
            # Mode "x" fails rather than overwrite if the file appeared since
            # the check above.
            pd.DataFrame({ROW_ID: row_ids}).to_csv(path, index=False, mode="x")
            written.append(path)
    return written


def load_training_set(stem: str, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the deduplicated training rows of one file.

    This is how preprocessing, the models and cross validation get their
    data. It reads the training indices only, never the test indices.

    Args:
        stem: File name without extension, e.g. 5year.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        The training rows as returned by load_file, in row_id order.

    Raises:
        FileNotFoundError: If the training indices have not been written yet.
    """
    if config is None:
        config = load_config()
    path = train_indices_path(stem, config)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(REPO_ROOT)} not found; run write_splits() once "
            "to freeze the split"
        )
    row_ids = pd.read_csv(path)[ROW_ID]
    return load_file(stem, config).loc[row_ids]


def cv_folds(
    stem: str, config: dict[str, Any] | None = None
) -> list[tuple[int, int, pd.Index, pd.Index]]:
    """Make the repeated stratified cross validation folds of one file.

    Every scorer uses these folds, so their out of fold predictions are
    paired. Only the training rows from load_training_set are split.

    Args:
        stem: File name without extension, e.g. 5year.
        config: Parsed project configuration, for the fold and repeat counts
            and the nominated seed. Read from config.yaml if None.

    Returns:
        One (repeat, fold, fit_row_ids, score_row_ids) tuple per fold, in
        order of repeat then fold, both numbered from 0. The row_ids are
        sorted, for selecting rows with .loc.
    """
    if config is None:
        config = load_config()
    frame = load_training_set(stem, config)
    n_folds = config["cv"]["n_folds"]
    splitter = RepeatedStratifiedKFold(
        n_splits=n_folds,
        n_repeats=config["cv"]["n_repeats"],
        random_state=config["seeds"]["nominated"],
    )

    folds = []
    # The splitter yields every fold of one repeat before the next repeat.
    for number, (fit, score) in enumerate(splitter.split(frame, frame[LABEL])):
        repeat, fold = divmod(number, n_folds)
        fit_row_ids = pd.Index(frame.index[fit], name=ROW_ID).sort_values()
        score_row_ids = pd.Index(frame.index[score], name=ROW_ID).sort_values()
        logger.info(
            "%s: repeat %d fold %d scores %d rows and %d bankrupt",
            stem,
            repeat,
            fold,
            len(score_row_ids),
            int(frame.loc[score_row_ids, LABEL].sum()),
        )
        folds.append((repeat, fold, fit_row_ids, score_row_ids))
    return folds


def inner_split(
    fit_rows: pd.DataFrame, config: dict[str, Any] | None = None
) -> tuple[pd.Index, pd.Index]:
    """Hold out early stopping rows from the fitting rows of one fold.

    Only the fitting rows are passed in, so early stopping never sees the
    fold being scored.

    Args:
        fit_rows: The fitting rows of one fold, indexed by row_id, with the
            class column.
        config: Parsed project configuration, for the inner validation
            fraction and the nominated seed. Read from config.yaml if None.

    Returns:
        The row_ids for fitting and the row_ids for early stopping, each
        sorted.
    """
    if config is None:
        config = load_config()
    fit, stop = train_test_split(
        fit_rows.index.to_numpy(),
        test_size=config["cv"]["inner_val_fraction"],
        random_state=config["seeds"]["nominated"],
        stratify=fit_rows[LABEL],
    )
    return (
        pd.Index(fit, name=ROW_ID).sort_values(),
        pd.Index(stop, name=ROW_ID).sort_values(),
    )
