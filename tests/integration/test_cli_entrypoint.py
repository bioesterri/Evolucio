"""Installed entry point and module execution smoke checks."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "command", [["evolucio", "--help"], [sys.executable, "-m", "evolucio.cli", "--version"]]
)
def test_installed_cli(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout
    assert not result.stderr


def test_dry_run_does_not_import_device_libraries() -> None:
    script = """
import sys
from evolucio.cli.main import main
assert main(['run', 'tests/fixtures/config/valid_v1.json',
             '--steps', '10', '--chunk-steps', '4', '--dry-run']) == 0
assert 'jax' not in sys.modules
assert 'equinox' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "chunks: 3" in result.stdout
    assert not result.stderr
