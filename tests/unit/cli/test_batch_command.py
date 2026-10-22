# ruff: noqa: ANN001, ANN201

import subprocess
import sys
from unittest.mock import Mock

import pytest

from evolucio.cli.main import main


def arguments(path):
    return [
        "batch",
        "run",
        "--config",
        f"A={path}",
        "--config",
        f"B={path}",
        "--seeds",
        "11,22",
        "--steps",
        "4",
        "--chunk-steps",
        "2",
    ]


def test_dry_run_and_real_output(config_path, monkeypatch, capsys):
    init = Mock(side_effect=AssertionError("must not initialize"))
    with monkeypatch.context() as patch:
        patch.setattr("evolucio.runtime.initialization.initialize_simulation", init)
        assert main([*arguments(config_path), "--dry-run"]) == 0
        init.assert_not_called()
    out = capsys.readouterr().out
    assert "cases: 2\nseeds: 2\nruns: 4" in out
    assert "001 A seed=11" in out and "004 B seed=22" in out
    assert "mode: experiment" in out and "steps/run: 4" in out
    assert main(arguments(config_path)) == 0
    output = capsys.readouterr()
    assert output.out.index("004 B seed=22") < output.out.index("[1/4]")
    assert output.out.count("... completed") == 4
    assert "runs requested: 4\nruns executed: 4\ncompleted: 4\nfailed: 0" in output.out
    assert not output.err


@pytest.mark.parametrize("seeds", ["", "1,", "1,,2", "true", "1.2", "1,1", "-1", "4294967296"])
def test_bad_seeds(config_path, seeds, capsys):
    args = arguments(config_path)
    args[args.index("--seeds") + 1] = seeds
    assert main(args) == 2
    assert capsys.readouterr().err


@pytest.mark.parametrize("config", ["no-equals", "=path.json", "A=", "A=missing.json"])
def test_bad_configs(config, capsys):
    assert (
        main(
            [
                "batch",
                "run",
                "--config",
                config,
                "--seeds",
                "1",
                "--steps",
                "2",
                "--chunk-steps",
                "1",
            ]
        )
        == 2
    )
    assert capsys.readouterr().err


@pytest.mark.parametrize("fail_fast, expected_runs", [(False, 4), (True, 1)])
def test_cli_technical_error_exit_one(config_path, monkeypatch, capsys, fail_fast, expected_runs):
    init = Mock(side_effect=RuntimeError("device failure"))
    monkeypatch.setattr("evolucio.runtime.initialization.initialize_simulation", init)
    args = arguments(config_path) + (["--fail-fast"] if fail_fast else [])
    assert main(args) == 1
    assert init.call_count == expected_runs
    output = capsys.readouterr()
    assert f"runs executed: {expected_runs}" in output.out
    assert f"failed: {expected_runs}" in output.out
    assert "A seed=11: RuntimeError: device failure" in output.err


def test_invalid_later_config_cli(config_path, tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    init = Mock()
    monkeypatch.setattr("evolucio.runtime.initialization.initialize_simulation", init)
    args = arguments(config_path)
    args[args.index(f"B={config_path}")] = f"B={bad}"
    assert main(args) == 2
    init.assert_not_called()
    assert capsys.readouterr().err


def test_installed_cli_dry_run_without_jax(config_path):
    script = """
import sys
from evolucio.cli.main import main
assert main(sys.argv[1:]) == 0
assert "jax" not in sys.modules
assert "equinox" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, *arguments(config_path), "--dry-run"], check=True)
