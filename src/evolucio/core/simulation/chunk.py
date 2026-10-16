"""Compiled execution of fixed-length simulation chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax

from evolucio.core.metrics import MetricsAccumulator, StepEventBuffer, StepMetrics
from evolucio.core.state import SimulationState

from .step import simulation_step

if TYPE_CHECKING:
    from evolucio.config.compile import CoreConfig


class RunChunkResult(eqx.Module):
    """Final carry and time-stacked compact outputs of one chunk."""

    state: SimulationState
    accumulator: MetricsAccumulator
    metrics: StepMetrics
    events: StepEventBuffer


def run_chunk_impl(
    *,
    state: SimulationState,
    accumulator: MetricsAccumulator,
    config: CoreConfig,
    chunk_steps: int,
) -> RunChunkResult:
    """Execute a positive, statically sized chunk with ``lax.scan``."""
    if isinstance(chunk_steps, bool) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        chunk_steps, int
    ):
        raise TypeError("chunk_steps must be a Python int, not bool")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be strictly positive")

    def scan_body(
        carry: tuple[SimulationState, MetricsAccumulator], _: None
    ) -> tuple[tuple[SimulationState, MetricsAccumulator], tuple[StepMetrics, StepEventBuffer]]:
        current_state, current_accumulator = carry
        result = simulation_step(
            state=current_state,
            accumulator=current_accumulator,
            config=config,
        )
        return (result.state, result.accumulator), (result.metrics, result.events)

    (final_state, final_accumulator), (metrics, events) = jax.lax.scan(
        scan_body,
        (state, accumulator),
        xs=None,
        length=chunk_steps,
        reverse=False,
        unroll=1,
    )
    return RunChunkResult(
        state=final_state,
        accumulator=final_accumulator,
        metrics=metrics,
        events=events,
    )


run_chunk = eqx.filter_jit(run_chunk_impl)
