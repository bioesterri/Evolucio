from pathlib import Path

import pytest

from evolucio.config import ExperimentConfig, load_config


@pytest.fixture
def config() -> ExperimentConfig:
    return load_config(Path("tests/fixtures/config/valid_v1.yaml"))
