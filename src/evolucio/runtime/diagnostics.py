"""Host invariant inspection and checked-step execution; no ecological rules."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np

from evolucio.core.diagnostics import (
    StateInvariantCode,
    StateInvariantReport,
    TransitionInvariantCode,
    check_final_state_invariants,
    check_structural_invariants,
)
from evolucio.core.simulation import simulation_step_checked as _simulation_step_checked
from evolucio.host.dto import HostArray

from .modes import (
    DebugFailure,
    DebugStepDiagnostic,
    InvariantDiagnostic,
    ModeRunResult,
    RunDiagnosticError,
    RunMode,
    SmokeReport,
)
from .runner import run_simulation
from .types import ChunkConsumer, HostChunkBatch, RunPlan, RunRuntimeResult

if TYPE_CHECKING:
    from evolucio.config.compile import CoreConfig
    from evolucio.core.metrics import MetricsAccumulator
    from evolucio.core.state import SimulationState

simulation_step_checked = eqx.filter_jit(_simulation_step_checked)


def _inspect(report: StateInvariantReport) -> InvariantDiagnostic:
    host = jax.device_get(report)
    return InvariantDiagnostic(bool(host.ok), tuple(int(count) for count in host.violation_counts))


def _require(
    diagnostic: InvariantDiagnostic,
    *,
    mode: RunMode,
    step: int,
    phase: str,
    codes: type[IntEnum] = StateInvariantCode,
) -> None:
    if not diagnostic.ok:
        violations = tuple(
            (code.name, diagnostic.violation_counts[int(code)])
            for code in codes
            if diagnostic.violation_counts[int(code)] > 0
        )
        raise RunDiagnosticError(
            mode,
            DebugFailure(
                step,
                phase,
                tuple(code for code, _ in violations),
                tuple(count for _, count in violations),
            ),
        )


def _initial(state: SimulationState, config: CoreConfig, mode: RunMode) -> InvariantDiagnostic:
    report = _inspect(
        check_structural_invariants(
            state=state,
            resource_capacity=config.world.resource_capacity,
            width=config.world.width,
            height=config.world.height,
        )
    )
    _require(report, mode=mode, step=int(state.step), phase="initial")
    return report


def run_smoke(
    *,
    state: SimulationState,
    accumulator: MetricsAccumulator,
    config: CoreConfig,
    plan: RunPlan,
    consumer: ChunkConsumer | None = None,
) -> ModeRunResult:
    """Check only run boundaries around the unchanged fast runtime."""
    initial = _initial(state, config, RunMode.SMOKE)
    runtime = run_simulation(
        state=state, accumulator=accumulator, config=config, plan=plan, consumer=consumer
    )
    final = _inspect(
        check_final_state_invariants(
            state=runtime.state,
            resource_capacity=config.world.resource_capacity,
            death_energy_threshold=config.energy.death_threshold,
            maximum_age=config.evolution.max_age,
            width=config.world.width,
            height=config.world.height,
        )
    )
    _require(final, mode=RunMode.SMOKE, step=runtime.final_step, phase="final")
    return ModeRunResult(RunMode.SMOKE, runtime, smoke_report=SmokeReport(initial, final))


def _temporal(value: HostArray) -> HostArray:
    """Add the single-step time axis without moving host arrays to a device."""
    return np.expand_dims(value, 0)


def run_debug(
    *,
    state: SimulationState,
    accumulator: MetricsAccumulator,
    config: CoreConfig,
    plan: RunPlan,
    consumer: ChunkConsumer | None = None,
) -> ModeRunResult:
    """Inspect every checked transition and fail before dispatching another step."""
    _initial(state, config, RunMode.DEBUG)
    initial_step = int(state.step)
    current_state, current_accumulator = state, accumulator
    latest = None
    for offset in range(plan.total_steps):
        step = initial_step + offset
        checked = simulation_step_checked(
            state=current_state, accumulator=current_accumulator, config=config
        )
        initial, final, transition = jax.device_get(
            (
                checked.initial_invariants,
                checked.final_invariants,
                checked.transition_invariants,
            )
        )
        latest = DebugStepDiagnostic(
            step,
            bool(initial.ok),
            bool(final.ok),
            bool(transition.ok),
            tuple(int(value) for value in initial.violation_counts),
            tuple(int(value) for value in final.violation_counts),
            tuple(int(value) for value in transition.violation_counts),
        )
        for phase, ok, counts, codes in (
            ("initial", latest.initial_ok, latest.initial_violation_counts, StateInvariantCode),
            ("final", latest.final_ok, latest.final_violation_counts, StateInvariantCode),
            (
                "transition",
                latest.transition_ok,
                latest.transition_violation_counts,
                TransitionInvariantCode,
            ),
        ):
            _require(
                InvariantDiagnostic(ok, counts),
                mode=RunMode.DEBUG,
                step=step,
                phase=phase,
                codes=codes,
            )
        result = checked.result
        metrics, events = jax.device_get((result.metrics, result.events))
        if consumer is not None:
            consumer(
                HostChunkBatch(
                    chunk_index=offset,
                    start_step=step,
                    end_step_exclusive=step + 1,
                    metrics=jax.tree.map(_temporal, metrics),
                    events=jax.tree.map(_temporal, events),
                )
            )
        current_state, current_accumulator = result.state, result.accumulator
    final_step = int(current_state.step)
    if final_step != initial_step + plan.total_steps:
        raise ValueError("final state.step does not match the requested horizon")
    return ModeRunResult(
        RunMode.DEBUG,
        RunRuntimeResult(
            current_state,
            current_accumulator,
            initial_step,
            final_step,
            plan.total_steps,
            plan.total_steps,
        ),
        latest_diagnostic=latest,
    )
