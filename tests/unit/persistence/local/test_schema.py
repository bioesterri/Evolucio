# ruff: noqa: ANN001, ANN201, ANN202

import subprocess
import sys
from unittest.mock import Mock

from evolucio.persistence.local import (
    RUN_MANIFEST_SCHEMA_NAME,
    RUN_MANIFEST_SCHEMA_VERSION,
    version_info,
)


def test_version_and_device_independence():
    assert RUN_MANIFEST_SCHEMA_NAME == "local_run_manifest_v1"
    assert RUN_MANIFEST_SCHEMA_VERSION == 1
    script = """
import sys
from evolucio.persistence.local import LocalArtifactRegistry, load_run_manifest
assert "jax" not in sys.modules and "equinox" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_git_version_success_and_failures(monkeypatch, no_git_calls):
    module = version_info
    runner = Mock(side_effect=[Mock(stdout="a" * 40 + "\n"), Mock(stdout=" M code.py\n")])
    monkeypatch.setattr(module.subprocess, "run", runner)
    info = no_git_calls()
    assert info.git_commit == "a" * 40 and info.git_dirty is True
    assert all(call.kwargs["timeout"] == 2 for call in runner.call_args_list)
    runner.side_effect = FileNotFoundError()
    assert no_git_calls().git_commit is None
    runner.side_effect = [Mock(stdout="a" * 40), subprocess.TimeoutExpired("git", 2)]
    assert no_git_calls().git_dirty is None


def test_missing_distribution(monkeypatch):
    def missing(name):
        raise version_info.PackageNotFoundError(name)

    monkeypatch.setattr(version_info, "version", missing)
    info = version_info.environment_info()
    assert info.python_version and info.platform
    assert info.jax_version is None and info.equinox_version is None
