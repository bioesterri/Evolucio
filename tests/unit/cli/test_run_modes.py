# ruff: noqa: ANN001, ANN201

from unittest.mock import Mock

import pytest

from evolucio.cli.main import main
from evolucio.runtime import DebugFailure, RunDiagnosticError, RunMode, get_mode_policy
from tests.unit.cli.test_run_command import summary


@pytest.mark.parametrize("mode", list(RunMode))
def test_dry_run_policy_and_no_execution(mode, config_path, monkeypatch, capsys):
    runtime = Mock()
    monkeypatch.setattr("evolucio.runtime.run_with_mode", runtime)
    assert (
        main(
            [
                "run",
                str(config_path),
                "--mode",
                mode.value,
                "--steps",
                "10",
                "--chunk-steps",
                "4",
                "--dry-run",
            ]
        )
        == 0
    )
    data = summary(capsys.readouterr().out)
    policy = get_mode_policy(mode)
    assert data["mode"] == mode.value
    assert data["benchmark_timing"] == policy.measure_runtime
    assert data["diagnostics_policy"] == {
        "per_step": policy.per_step_diagnostics,
        "initial": policy.initial_invariant_check,
        "final": policy.final_invariant_check,
    }
    assert data["chunks"] == 3 and data["remainder"] == 2
    runtime.assert_not_called()


def test_default_and_invalid_mode(config_path, capsys):
    assert main(["run", str(config_path), "--steps", "1", "--dry-run"]) == 0
    assert summary(capsys.readouterr().out)["mode"] == "experiment"
    assert main(["run", str(config_path), "--steps", "1", "--mode", "invalid"]) == 2
    assert capsys.readouterr().err


@pytest.mark.parametrize("mode", [RunMode.SMOKE, RunMode.DEBUG, RunMode.BENCHMARK])
def test_cli_modes_complete(mode, config_path, capsys):
    assert (
        main(["run", str(config_path), "--mode", mode.value, "--steps", "2", "--chunk-steps", "1"])
        == 0
    )
    output = capsys.readouterr()
    data = summary(output.out)
    assert data["mode"] == mode.value
    assert data["status"] == "completed"
    assert data["final_step"] == 2
    assert not output.err
    if mode is RunMode.SMOKE:
        assert data["initial_invariants"] == data["final_invariants"] == "ok"
    if mode is RunMode.BENCHMARK:
        assert data["warmup"] == "completed"
        assert data["elapsed_seconds"] > 0
        assert data["steps_per_second"] > 0
        assert data["platform"]


@pytest.mark.parametrize("mode", [RunMode.SMOKE, RunMode.DEBUG])
def test_quiet_does_not_hide_failure(mode, config_path, monkeypatch, capsys):
    failure = RunDiagnosticError(
        mode, DebugFailure(7, "transition", ("POPULATION_BALANCE_MISMATCH",), (1,))
    )
    monkeypatch.setattr("evolucio.runtime.run_with_mode", Mock(side_effect=failure))
    assert main(["run", str(config_path), "--mode", mode.value, "--steps", "10", "--quiet"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "status: failed" in output.err
    assert "step: 7" in output.err
    assert "phase: transition" in output.err
    assert "POPULATION_BALANCE_MISMATCH: 1" in output.err
    assert "Traceback" not in output.err


def test_benchmark_quiet_preserves_timing(config_path, capsys):
    assert main(["run", str(config_path), "--mode", "benchmark", "--steps", "1", "--quiet"]) == 0
    assert summary(capsys.readouterr().out)["elapsed_seconds"] > 0


def test_invalid_configuration_never_dispatches(config_path, monkeypatch, capsys):
    runtime = Mock()
    monkeypatch.setattr("evolucio.runtime.run_with_mode", runtime)
    config_path.write_text("{}")
    assert main(["run", str(config_path), "--mode", "debug", "--steps", "1"]) == 2
    assert capsys.readouterr().err
    runtime.assert_not_called()
