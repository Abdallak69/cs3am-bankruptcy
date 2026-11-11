"""Repository root and project configuration.

Every path in the project resolves from REPO_ROOT, never from the working
directory, so code behaves the same from a notebook, a test or the shell.

Owner: Abdalla (project base).
"""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = REPO_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read and parse the project configuration file.

    Args:
        path: Location of the YAML file. Defaults to config.yaml in the
            repository root.

    Returns:
        The parsed configuration as a nested dictionary. Paths inside it are
        relative to REPO_ROOT and are left as strings for the caller to join.
    """
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
