"""config.yaml loads and points at directories that exist."""

from src.config import REPO_ROOT, load_config


def test_config_loads() -> None:
    config = load_config()
    assert isinstance(config, dict)
    assert config


def test_configured_directories_exist() -> None:
    paths = load_config()["paths"]
    directories = {key: value for key, value in paths.items() if key.endswith("_dir")}

    assert directories
    missing = [
        key for key, value in directories.items() if not (REPO_ROOT / value).is_dir()
    ]
    assert missing == []


def test_42_is_a_model_seed() -> None:
    assert 42 in load_config()["seeds"]["models"]
