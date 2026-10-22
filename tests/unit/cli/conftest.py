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


@pytest.fixture(autouse=True)
def isolated_recording(tmp_path, monkeypatch):
    from evolucio.persistence.local.version_info import CodeVersionInfo

    monkeypatch.setattr(
        "evolucio.persistence.local.version_info.code_version_info",
        lambda: CodeVersionInfo("0.1.0.dev0", None, None),
    )
    monkeypatch.setattr("evolucio.cli.run_command.DEFAULT_RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr("evolucio.cli.batch_command.DEFAULT_RUNS_ROOT", tmp_path / "runs")
