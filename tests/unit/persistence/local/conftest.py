# ruff: noqa: ANN001, ANN201

import pytest

from evolucio.config import load_config
from evolucio.persistence.local import create_local_run, version_info
from evolucio.persistence.local.version_info import CodeVersionInfo
from evolucio.runtime import RunMode, RunPlan


@pytest.fixture(autouse=True)
def no_git_calls(monkeypatch):
    original = version_info.code_version_info
    monkeypatch.setattr(
        "evolucio.persistence.local.version_info.code_version_info",
        lambda: CodeVersionInfo("0.1.0.dev0", "a" * 40, False),
    )

    return original


@pytest.fixture
def host_config():
    return load_config("tests/fixtures/config/valid_v1.json")


@pytest.fixture
def recorder(tmp_path, host_config):
    return create_local_run(
        config=host_config,
        plan=RunPlan(4, 2),
        mode=RunMode.EXPERIMENT,
        output_root=tmp_path / "runs",
    )
