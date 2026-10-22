"""Host boundaries around existing modes; no scientific stepping logic is duplicated."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from evolucio.persistence.local import LocalRunRecorder
from evolucio.persistence.zarr.checkpoint import write_checkpoint
from evolucio.persistence.zarr.index import ZarrRunIndex
from evolucio.persistence.zarr.schedule import next_chunk_length
from evolucio.persistence.zarr.schemas import ZarrCapturePolicy
from evolucio.persistence.zarr.snapshot import write_snapshot

from .modes import ModeRunResult, RunMode, SmokeReport, run_with_mode
from .snapshot import simulation_state_to_snapshot
from .types import ChunkConsumer, HostChunkBatch, RunPlan, RunRuntimeResult

if TYPE_CHECKING:
    from evolucio.config.compile import CoreConfig
    from evolucio.core.metrics import MetricsAccumulator
    from evolucio.core.state import SimulationState


class RunCaptures:
    """Own one run's index, explicit capture policy and host capture context."""

    def __init__(
        self, recorder: LocalRunRecorder, config: CoreConfig, policy: ZarrCapturePolicy
    ) -> None:
        self.writer = ZarrRunIndex(recorder, policy)
        self.recorder = recorder
        self.config = config
        self.policy = policy

    def capture(
        self, state: SimulationState, accumulator: MetricsAccumulator, *, final: bool
    ) -> None:
        step = int(state.step)
        if (
            self.policy.checkpoint_interval is not None
            and step % self.policy.checkpoint_interval == 0
        ) or (final and self.policy.checkpoint_on_completion):
            record = write_checkpoint(
                state=state,
                accumulator=accumulator,
                config=self.config,
                run_dir=self.recorder.run_dir,
                run_id=self.recorder.run_id,
                config_hash=self.recorder.manifest.config_hash,
                compile_signature=self.writer.index.compile_signature,
            )
            self.writer.add(record)
        if (
            self.policy.snapshot_interval is not None and step % self.policy.snapshot_interval == 0
        ) or (final and self.policy.snapshot_on_completion):
            snapshot = write_snapshot(
                snapshot=simulation_state_to_snapshot(state),
                run_dir=self.recorder.run_dir,
                run_id=self.recorder.run_id,
                config_hash=self.recorder.manifest.config_hash,
                compile_signature=self.writer.index.compile_signature,
            )
            self.writer.add(snapshot)


def run_with_captures(
    *,
    mode: RunMode,
    state: SimulationState,
    accumulator: MetricsAccumulator,
    config: CoreConfig,
    plan: RunPlan,
    captures: RunCaptures,
    consumer: ChunkConsumer | None = None,
) -> ModeRunResult:
    initial_step = int(state.step)
    chunks = 0
    emitted = 0
    remaining = plan.total_steps
    first: ModeRunResult | None = None
    last: ModeRunResult | None = None
    elapsed = 0.0

    def reindexed(batch: HostChunkBatch) -> None:
        nonlocal emitted
        if consumer is not None:
            consumer(replace(batch, chunk_index=emitted))
        emitted += 1

    while remaining:
        length = next_chunk_length(
            state_step=int(state.step),
            remaining_steps=remaining,
            chunk_steps=plan.chunk_steps,
            policy=captures.policy,
        )
        last = run_with_mode(
            mode=mode,
            state=state,
            accumulator=accumulator,
            config=config,
            plan=RunPlan(length, length),
            consumer=reindexed,
        )
        if first is None:
            first = last
        state, accumulator = last.runtime.state, last.runtime.accumulator
        chunks += last.runtime.chunks_executed
        remaining -= length
        if last.benchmark is not None:
            elapsed += last.benchmark.elapsed_seconds
        captures.capture(state, accumulator, final=remaining == 0)
    if first is None or last is None:
        raise ValueError("capture execution requires a nonempty plan")
    runtime = RunRuntimeResult(
        state, accumulator, initial_step, int(state.step), plan.total_steps, chunks
    )
    smoke = None
    if first.smoke_report is not None and last.smoke_report is not None:
        smoke = SmokeReport(first.smoke_report.initial, last.smoke_report.final)
    benchmark = last.benchmark
    if benchmark is not None:
        benchmark = replace(
            benchmark,
            total_steps=plan.total_steps,
            chunks_executed=chunks,
            initial_step=initial_step,
            elapsed_seconds=elapsed,
            steps_per_second=plan.total_steps / elapsed,
        )
    return ModeRunResult(
        mode,
        runtime,
        smoke_report=smoke,
        latest_diagnostic=last.latest_diagnostic,
        benchmark=benchmark,
    )
