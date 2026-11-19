"""Only src/split.py (writes once) and src/evaluate.py (reads) may refer to the
test indices, or to split_file, which can rebuild them from the raw data. No
other module and no notebook may refer to either."""

from pathlib import Path

from src.config import REPO_ROOT, load_config

TEST_INDICES_KEY = "test_indices_pattern"
ALLOWED_FILES = {"src/split.py", "src/evaluate.py"}


def test_no_other_module_or_notebook_references_test_indices() -> None:
    pattern = load_config()["paths"][TEST_INDICES_KEY]
    # Also catch the fixed file name ending, in case the path is rebuilt from
    # parts rather than copied whole.
    fixed_ending = Path(pattern).name.rsplit("}", 1)[-1]
    needles = {TEST_INDICES_KEY, pattern, fixed_ending, "split_file"}

    sources = sorted((REPO_ROOT / "src").rglob("*.py")) + sorted(
        (REPO_ROOT / "notebooks").rglob("*.ipynb")
    )
    offenders = []
    for source in sources:
        relative = source.relative_to(REPO_ROOT).as_posix()
        if relative in ALLOWED_FILES:
            continue
        text = source.read_text(encoding="utf-8")
        offenders += [f"{relative}: {needle}" for needle in needles if needle in text]

    assert offenders == []
