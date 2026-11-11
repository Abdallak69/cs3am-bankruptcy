# Instructions for AI assistants

These rules come from the project plan and apply to every change in this
repository. Every member must be able to explain every line they commit,
including GenAI assisted code, so keep changes small and readable.

## Code standards

- Follow PEP 8 and format with black (configured in `pyproject.toml`). Every
  function in `src/` has a docstring and type hints.
- Seeds and paths come from `config.yaml` via `src/config.py`. Never hard code
  a seed or a path; resolve paths from `REPO_ROOT`, never from the working
  directory.
- Notebooks import from `src/`. No logic lives only in a notebook.
- Borrowed code carries a source comment.
- Use British English in comments, docstrings and documentation.

## Leakage rules

- Only `src/evaluate.py` may read the test indices. `src/split.py` writes them
  once; no other module touches them.
- Every transform, calibrator and threshold is fitted on training data only
  (inside folds or on out of fold predictions), never on the folds being
  scored or on test data.
- The five horizon files are never pooled.

## Commits and pull requests

- Never add AI attribution to commit messages, pull requests or code: no
  `Co-authored-by` trailers, "Made with" or "Generated with" lines for Cursor,
  Claude or any other AI tool. Commits are authored by the team member who
  makes them.
