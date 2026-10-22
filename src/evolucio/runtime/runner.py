"""Synchronous host orchestration around the compiled run_chunk kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from evolucio.core.metrics import MetricsAccumulator
from evolucio.core.simulation import run_chunk
from evolucio.core.state import SimulationState

from .types import ChunkConsumer, HostChunkBatch, RunPlan, RunRuntimeResult

if TYPE_CHECKING:
    from evolucio.config.compile import CoreConfig


def run_simulation(
    *,
    state: SimulationState,
    accumulator: MetricsAccumulator,
    config: CoreConfig,
    plan: RunPlan,
    consumer: ChunkConsumer | None = None,
) -> RunRuntimeResult:
    """Execute exactly the horizon, streaming all metrics and events per chunk.

    Only the initial/final step scalars are read from the carry. State and
    accumulator are passed directly between compiled chunks. No extinction
    check or history retention changes the requested execution horizon.
    """
    initial_step = int(state.step)
    current_state = state
    current_accumulator = accumulator
    full_chunks, remainder = divmod(plan.total_steps, plan.chunk_steps)
    chunks_executed = full_chunks + int(remainder > 0)
    start_step = initial_step

    for chunk_index in range(chunks_executed):
        length = plan.chunk_steps if chunk_index < full_chunks else remainder
        chunk = run_chunk(
            state=current_state,
            accumulator=current_accumulator,
            config=config,
            chunk_steps=length,
        )
        current_state = chunk.state
        current_accumulator = chunk.accumulator
        host_metrics, host_events = jax.device_get((chunk.metrics, chunk.events))
        end_step = start_step + length
        if host_metrics.step.shape != (length,) or any(
            int(step) != start_step + offset for offset, step in enumerate(host_metrics.step)
        ):
            raise ValueError("metrics.step does not match the planned chunk interval")
        if consumer is not None:
            consumer(
                HostChunkBatch(
                    chunk_index=chunk_index,
                    start_step=start_step,
                    end_step_exclusive=end_step,
                    metrics=host_metrics,
                    events=host_events,
                )
            )
        start_step = end_step
        # Release outputs before dispatching the next chunk, even without a sink.
        del chunk, host_metrics, host_events

    final_step = int(current_state.step)
    if final_step != initial_step + plan.total_steps:
        raise ValueError("final state.step does not match the requested horizon")
    return RunRuntimeResult(
        state=current_state,
        accumulator=current_accumulator,
        initial_step=initial_step,
        final_step=final_step,
        steps_executed=plan.total_steps,
        chunks_executed=chunks_executed,
    )
