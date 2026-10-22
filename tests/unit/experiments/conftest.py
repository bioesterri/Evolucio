# ruff: noqa: ANN001, ANN201

import json
from pathlib import Path

import pytest


@pytest.fixture
def config_path(tmp_path):
    data = json.loads(Path("tests/fixtures/config/valid_v1.json").read_text())
    data["world"].update(width=4, height=4, resource_distribution="uniform")
    data["population"].update(initial_agents=2, max_agents=4, max_births_per_step=2)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(data))
    return path
