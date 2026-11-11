"""Only src/split.py (writes once) and src/evaluate.py (reads) may refer to the
test indices."""

from pathlib import Path

from src.config import REPO_ROOT, load_config

TEST_INDICES_KEY = "test_indices_pattern"
ALLOWED_FILES = {"split.py", "evaluate.py"}


def test_no_other_src_module_references_test_indices() -> None:
    pattern = load_config()["paths"][TEST_INDICES_KEY]
    # Also catch the fixed file name ending, in case the path is rebuilt from
    # parts rather than copied whole.
    fixed_ending = Path(pattern).name.rsplit("}", 1)[-1]
    needles = {TEST_INDICES_KEY, pattern, fixed_ending}

    offenders = []
    for source in sorted((REPO_ROOT / "src").rglob("*.py")):
        if source.name in ALLOWED_FILES:
            continue
        text = source.read_text(encoding="utf-8")
        offenders += [
            f"{source.relative_to(REPO_ROOT)}: {needle}"
            for needle in needles
            if needle in text
        ]

    assert offenders == []
