# ruff: noqa: ANN001, ANN201

import json
from pathlib import Path

import pytest


@pytest.fixture
def config_data():
    data = json.loads(Path("tests/fixtures/config/valid_v1.json").read_text())
    data["world"].update(width=3, height=3, resource_distribution="uniform")
    data["population"].update(initial_agents=2, max_agents=3, max_births_per_step=1)
    data["runtime"].update(chunk_size=4)
    return data


@pytest.fixture
def config_path(tmp_path, config_data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_data))
    return path
