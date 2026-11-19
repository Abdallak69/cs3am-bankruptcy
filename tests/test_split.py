"""split.py makes one stratified split per file, writes it once, and serves the
training rows without reading the test indices."""

from pathlib import Path

import pandas as pd
import pytest

import src.split as split
from src.config import load_config
from src.load import LABEL, load_file

STEMS = ["1year", "2year", "3year", "4year", "5year"]
CONFIG = load_config()


@pytest.fixture(scope="module")
def files() -> dict[str, pd.DataFrame]:
    return {stem: load_file(stem, CONFIG) for stem in STEMS}


@pytest.fixture
def scratch_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Writes go to an empty temporary root, never to the frozen data/splits/.
    monkeypatch.setattr(split, "REPO_ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize("stem", STEMS)
def test_parts_disjoint_and_cover_file(
    files: dict[str, pd.DataFrame], stem: str
) -> None:
    train, test = split.split_file(files[stem], stem, CONFIG)
    assert train.intersection(test).empty
    assert train.union(test).equals(files[stem].index)


@pytest.mark.parametrize("stem", STEMS)
def test_bankrupt_share_within_one_row(
    files: dict[str, pd.DataFrame], stem: str
) -> None:
    frame = files[stem]
    share = frame[LABEL].mean()
    for row_ids in split.split_file(frame, stem, CONFIG):
        bankrupt = frame.loc[row_ids, LABEL].sum()
        assert abs(bankrupt - share * len(row_ids)) <= 1


@pytest.mark.parametrize("stem", STEMS)
def test_same_seed_gives_saved_split(files: dict[str, pd.DataFrame], stem: str) -> None:
    # Training and test partition the file, so matching training indices
    # also fix the test indices without this test reading them.
    saved = pd.read_csv(split.train_indices_path(stem, CONFIG))
    train, _ = split.split_file(files[stem], stem, CONFIG)

    assert list(saved.columns) == ["row_id"]
    assert saved["row_id"].is_monotonic_increasing
    assert saved["row_id"].tolist() == train.tolist()


def test_writing_twice_raises(scratch_root: Path) -> None:
    written = split.write_splits(CONFIG)
    contents = {path: path.read_bytes() for path in written}
    assert len(written) == 10

    with pytest.raises(FileExistsError, match="split is frozen"):
        split.write_splits(CONFIG)
    assert {path: path.read_bytes() for path in written} == contents


def test_load_training_set_returns_only_training_rows(
    files: dict[str, pd.DataFrame], scratch_root: Path
) -> None:
    split.write_splits(CONFIG)
    # Deleting the test indices shows load_training_set never needs them.
    for stem in STEMS:
        pattern = CONFIG["paths"]["test_indices_pattern"]
        (scratch_root / pattern.format(stem=stem)).unlink()

    for stem in STEMS:
        train, test = split.split_file(files[stem], stem, CONFIG)
        loaded = split.load_training_set(stem, CONFIG)
        pd.testing.assert_frame_equal(loaded, files[stem].loc[train])
        assert loaded.index.intersection(test).empty
