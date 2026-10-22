# ruff: noqa: ANN001, ANN201

from unittest.mock import Mock

import equinox as eqx
import pytest

from evolucio.core.diagnostics import check_step_transition_invariants
from evolucio.core.metrics import create_empty_metrics_accumulator
from evolucio.core.simulation.step import CheckedSimulationStepResult
from evolucio.runtime import RunDiagnosticError, RunMode, RunPlan, run_with_mode
from tests.regression.helpers import canonical_case


def test_debug_checked_step_fail_fast(monkeypatch):
    from evolucio.runtime import diagnostics

    original = diagnostics.simulation_step_checked
    calls = []

    def corrupt(*, state, accumulator, config) -> CheckedSimulationStepResult:
        calls.append(int(state.step))
        checked = original(state=state, accumulator=accumulator, config=config)
        if int(state.step) == 1:
            metrics = eqx.tree_at(
                lambda m: m.alive_count,
                checked.result.metrics,
                checked.result.metrics.alive_count + 1,
            )
            transition = check_step_transition_invariants(
                before=state,
                after=checked.result.state,
                metrics=metrics,
                events=checked.result.events,
                accumulator=checked.result.accumulator,
            )
            checked = eqx.tree_at(lambda c: c.transition_invariants, checked, transition)
        return checked

    monkeypatch.setattr(diagnostics, "simulation_step_checked", corrupt)
    state, config, _ = canonical_case()
    consumer = Mock()
    with pytest.raises(RunDiagnosticError) as failure:
        run_with_mode(
            mode=RunMode.DEBUG,
            state=state,
            accumulator=create_empty_metrics_accumulator(),
            config=config,
            plan=RunPlan(5, 3),
            consumer=consumer,
        )
    assert calls == [0, 1]
    assert consumer.call_count == 1
    assert failure.value.failure.step == 1
    assert failure.value.failure.phase == "transition"
    assert failure.value.failure.invariant_codes == ("METRIC_ALIVE_COUNT_MISMATCH",)
    assert failure.value.failure.violation_counts == (1,)
    assert "METRIC_ALIVE_COUNT_MISMATCH: 1" in str(failure.value)


def test_initial_corruption_prevents_checked_step(monkeypatch):
    checked = Mock()
    monkeypatch.setattr("evolucio.runtime.diagnostics.simulation_step_checked", checked)
    state, config, _ = canonical_case()
    bad = eqx.tree_at(lambda s: s.world.occupancy, state, state.world.occupancy.at[0, 0].set(-1))
    with pytest.raises(RunDiagnosticError, match="OCCUPANCY_NEGATIVE"):
        run_with_mode(
            mode=RunMode.DEBUG,
            state=bad,
            accumulator=create_empty_metrics_accumulator(),
            config=config,
            plan=RunPlan(3, 2),
        )
    checked.assert_not_called()
