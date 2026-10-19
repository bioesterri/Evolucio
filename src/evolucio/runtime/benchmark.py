"""Timing of the normal host runtime after discarded, synchronized warmup."""

from __future__ import annotations

from math import isfinite
from time import perf_counter
from typing import TYPE_CHECKING

import jax

from evolucio.core.simulation import run_chunk

from .modes import BenchmarkResult, ModeRunResult, RunMode
from .runner import run_simulation
from .types import RunPlan

if TYPE_CHECKING:
    from evolucio.config.compile import CoreConfig
    from evolucio.core.metrics import MetricsAccumulator
    from evolucio.core.state import SimulationState


def run_benchmark(
    *,
    state: SimulationState,
    accumulator: MetricsAccumulator,
    config: CoreConfig,
    plan: RunPlan,
) -> ModeRunResult:
    """Measure host + kernel + required host boundary, without DTO consumers."""
    full_chunks, remainder = divmod(plan.total_steps, plan.chunk_steps)
    lengths = ((plan.chunk_steps,) if full_chunks else ()) + ((remainder,) if remainder else ())
    for length in lengths:
        warmup = run_chunk(state=state, accumulator=accumulator, config=config, chunk_steps=length)
        jax.block_until_ready(warmup)  # pyright: ignore[reportUnknownMemberType]
        del warmup
    jax.block_until_ready((state, accumulator))  # pyright: ignore[reportUnknownMemberType]
    device = next(iter(state.step.devices()))
    start = perf_counter()
    runtime = run_simulation(
        state=state, accumulator=accumulator, config=config, plan=plan, consumer=None
    )
    jax.block_until_ready((runtime.state, runtime.accumulator))  # pyright: ignore[reportUnknownMemberType]
    elapsed = perf_counter() - start
    if not isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("benchmark elapsed_seconds must be finite and strictly positive")
    return ModeRunResult(
        RunMode.BENCHMARK,
        runtime,
        benchmark=BenchmarkResult(
            total_steps=plan.total_steps,
            chunks_executed=runtime.chunks_executed,
            elapsed_seconds=elapsed,
            steps_per_second=plan.total_steps / elapsed,
            initial_step=runtime.initial_step,
            final_step=runtime.final_step,
            platform=device.platform,
            device_kind=device.device_kind,
        ),
    )
