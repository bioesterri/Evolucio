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


@pytest.mark.parametrize("mode", ["smoke", "debug", "experiment", "benchmark"])
def test_dry_run_does_not_import_device_libraries(mode: str) -> None:
    script = """
import sys
from evolucio.cli.main import main
assert main(['run', 'tests/fixtures/config/valid_v1.json',
             '--steps', '10', '--chunk-steps', '4', '--mode', sys.argv[1], '--dry-run']) == 0
assert 'jax' not in sys.modules
assert 'equinox' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, mode], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "chunks: 3" in result.stdout
    assert not result.stderr
