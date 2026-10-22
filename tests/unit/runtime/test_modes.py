# ruff: noqa: ANN001, ANN201

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import equinox as eqx
import pytest

from evolucio.runtime import (
    RunDiagnosticError,
    RunMode,
    RunPlan,
    RunRuntimeResult,
    get_mode_policy,
    run_with_mode,
)
from tests.regression.helpers import canonical_case


@pytest.mark.parametrize("mode", list(RunMode))
def test_policies(mode):
    policy = get_mode_policy(RunMode(mode.value))
    assert policy.mode is mode
    assert policy.collect_telemetry == (mode is not RunMode.BENCHMARK)
    assert policy.per_step_diagnostics == (mode is RunMode.DEBUG)
    assert policy.initial_invariant_check == (mode in (RunMode.SMOKE, RunMode.DEBUG))
    assert policy.final_invariant_check == policy.initial_invariant_check
    assert policy.measure_runtime == policy.warmup_before_timing == (mode is RunMode.BENCHMARK)
    with pytest.raises(FrozenInstanceError):
        policy.show_progress = True


def test_invalid_mode():
    with pytest.raises(ValueError):
        RunMode("unknown")


def test_experiment_delegates(monkeypatch):
    state, accumulator, config, consumer = object(), object(), object(), Mock()
    plan = RunPlan(5, 2)
    runtime = Mock(return_value=object())
    monkeypatch.setattr("evolucio.runtime.run_simulation", runtime)
    result = run_with_mode(
        mode=RunMode.EXPERIMENT,
        state=state,
        accumulator=accumulator,
        config=config,
        plan=plan,
        consumer=consumer,
    )
    runtime.assert_called_once_with(
        state=state, accumulator=accumulator, config=config, plan=plan, consumer=consumer
    )
    assert result.runtime is runtime.return_value
    assert result.benchmark is result.smoke_report is result.latest_diagnostic is None


def test_smoke_checks_both_boundaries(monkeypatch):
    from evolucio.core.metrics import create_empty_metrics_accumulator
    from evolucio.runtime import diagnostics

    state, config, _ = canonical_case()
    accumulator = create_empty_metrics_accumulator()
    end = eqx.tree_at(lambda s: s.step, state, state.step + 1)
    runtime = Mock(return_value=RunRuntimeResult(end, accumulator, 0, 1, 1, 1))
    monkeypatch.setattr(diagnostics, "run_simulation", runtime)
    initial = Mock(wraps=diagnostics.check_structural_invariants)
    final = Mock(wraps=diagnostics.check_final_state_invariants)
    monkeypatch.setattr(diagnostics, "check_structural_invariants", initial)
    monkeypatch.setattr(diagnostics, "check_final_state_invariants", final)
    result = run_with_mode(
        mode=RunMode.SMOKE, state=state, accumulator=accumulator, config=config, plan=RunPlan(1, 1)
    )
    assert result.smoke_report.initial.ok and result.smoke_report.final.ok
    assert initial.call_count == final.call_count == runtime.call_count == 1


@pytest.mark.parametrize("phase", ["initial", "final"])
def test_smoke_detects_corruption(phase, monkeypatch):
    from evolucio.core.metrics import create_empty_metrics_accumulator

    state, config, _ = canonical_case()
    bad = eqx.tree_at(lambda s: s.world.resources, state, state.world.resources.at[0, 0].set(-1))
    accumulator = create_empty_metrics_accumulator()
    runtime = Mock(return_value=RunRuntimeResult(bad, accumulator, 0, 1, 1, 1))
    monkeypatch.setattr("evolucio.runtime.diagnostics.run_simulation", runtime)
    with pytest.raises(RunDiagnosticError) as failure:
        run_with_mode(
            mode=RunMode.SMOKE,
            state=bad if phase == "initial" else state,
            accumulator=accumulator,
            config=config,
            plan=RunPlan(1, 1),
        )
    assert failure.value.failure.phase == phase
    assert "RESOURCE_OUT_OF_RANGE" in failure.value.failure.invariant_codes
    assert failure.value.failure.violation_counts == (1,)
    assert runtime.call_count == (0 if phase == "initial" else 1)
    assert float(bad.world.resources[0, 0]) == -1
