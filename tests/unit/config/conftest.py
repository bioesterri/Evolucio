from pathlib import Path

import pytest

from evolucio.config import ExperimentConfig, load_config


@pytest.fixture
def valid_path() -> Path:
    return Path("tests/fixtures/config/valid_v1.yaml")


@pytest.fixture
def config(valid_path: Path) -> ExperimentConfig:
    return load_config(valid_path)
