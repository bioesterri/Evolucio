"""Host orchestration and bounded retention of run summary metrics."""

from dataclasses import dataclass
from pathlib import Path

import typer

from evolucio.config import FrozenConfig
from evolucio.config.errors import ConfigError
from evolucio.host import ChunkTelemetryDTO, StepMetricsDTO
from evolucio.persistence.local import DEFAULT_RUNS_ROOT, LocalRunRecorder
from evolucio.persistence.zarr.schedule import expected_capture_steps, next_chunk_length
from evolucio.persistence.zarr.schemas import DEFAULT_ZARR_POLICY, ZarrCapturePolicy
from evolucio.runtime import RunMode, RunPlan, get_mode_policy

from .config_commands import load_frozen
from .formatting import text_fields


@dataclass(slots=True)
class CliRunSummaryConsumer:
    """Retain only the last scalar record and cumulative event counts."""

    latest_metrics: StepMetricsDTO | None = None
    births_total: int = 0
    deaths_total: int = 0

    def __call__(self, batch: ChunkTelemetryDTO) -> None:
        for metrics in batch.metrics:
            self.latest_metrics = metrics
            self.births_total += metrics.birth_count
            self.deaths_total += metrics.death_count


def run_config(
    path: Path,
    *,
    steps: int,
    chunk_steps: int | None,
    quiet: bool,
    dry_run: bool,
    mode: RunMode = RunMode.EXPERIMENT,
    output_root: Path | None = None,
    capture_policy: ZarrCapturePolicy = DEFAULT_ZARR_POLICY,
) -> str:
    """Prepare a plan or execute through PR-30 and the PR-31 DTO adapter."""
    policy = get_mode_policy(mode)
    frozen = load_frozen(path)
    plan = RunPlan(
        total_steps=steps,
        chunk_steps=frozen.config.runtime.chunk_size if chunk_steps is None else chunk_steps,
    )
    if dry_run:
        full_chunks, remainder = divmod(plan.total_steps, plan.chunk_steps)
        chunks = full_chunks + int(remainder > 0)
        capture_plan: dict[str, object] = {}
        if capture_policy.enabled:
            position = 0
            lengths: list[int] = []
            while position < plan.total_steps:
                length = next_chunk_length(
                    state_step=position,
                    remaining_steps=plan.total_steps - position,
                    chunk_steps=plan.chunk_steps,
                    policy=capture_policy,
                )
                lengths.append(length)
                position += length
            chunks = len(lengths)
            full_chunks = lengths.count(plan.chunk_steps)
            remainder = lengths[-1] if lengths[-1] != plan.chunk_steps else 0
            capture_plan["planned_chunk_lengths"] = lengths
        return text_fields(
            {
                "checkpoint_every": capture_policy.checkpoint_interval,
                "snapshot_every": capture_policy.snapshot_interval,
                "expected_checkpoint_steps": expected_capture_steps(
                    initial_step=0,
                    total_steps=steps,
                    interval=capture_policy.checkpoint_interval,
                    on_completion=capture_policy.checkpoint_on_completion,
                ),
                "expected_snapshot_steps": expected_capture_steps(
                    initial_step=0,
                    total_steps=steps,
                    interval=capture_policy.snapshot_interval,
                    on_completion=capture_policy.snapshot_on_completion,
                ),
                "mode": mode.value,
                "diagnostics_policy": {
                    "per_step": policy.per_step_diagnostics,
                    "initial": policy.initial_invariant_check,
                    "final": policy.final_invariant_check,
                },
                "benchmark_timing": policy.measure_runtime,
                "warmup_before_timing": policy.warmup_before_timing,
                "config_path": str(path),
                "config_hash": frozen.config_hash,
                "seed": frozen.config.seed,
                "total_steps": plan.total_steps,
                "chunk_steps": plan.chunk_steps,
                "chunks": chunks,
                **capture_plan,
                "full_chunks": full_chunks,
                "remainder": remainder,
            }
        )
    return _execute(
        frozen,
        plan=plan,
        quiet=quiet,
        mode=mode,
        output_root=output_root,
        capture_policy=capture_policy,
    )


def _execute(
    frozen: FrozenConfig,
    *,
    plan: RunPlan,
    quiet: bool,
    mode: RunMode,
    output_root: Path | None = None,
    capture_policy: ZarrCapturePolicy = DEFAULT_ZARR_POLICY,
) -> str:
    """Import device-facing APIs only after validation and dry-run planning."""
    from evolucio.config.compile import ConfigCompilationError, compile_config
    from evolucio.runtime import DTOChunkConsumer
    from evolucio.runtime.recorded_run import run_recorded_simulation

    try:
        compiled = compile_config(frozen.config)
    except ConfigCompilationError as exc:
        raise ConfigError(str(exc)) from exc
    consumer = CliRunSummaryConsumer()
    recorder: LocalRunRecorder | None = None

    def created(value: LocalRunRecorder) -> None:
        nonlocal recorder
        recorder = value
        if not quiet:
            typer.echo(text_fields({"run_id": value.run_id, "run_dir": str(value.run_dir)}))

    try:
        recorded = run_recorded_simulation(
            config=frozen.config,
            compiled=compiled,
            capture_policy=capture_policy,
            plan=plan,
            mode=mode,
            output_root=DEFAULT_RUNS_ROOT if output_root is None else output_root,
            consumer=DTOChunkConsumer(consumer),
            on_created=created,
        )
    except Exception as exc:
        if recorder is not None:
            typer.echo(
                text_fields(
                    {
                        "status": recorder.manifest.status.value,
                        "run_id": recorder.run_id,
                        "manifest": str(recorder.run_dir / "manifest.json"),
                    }
                ),
                err=True,
            )
        for note in getattr(exc, "__notes__", ()):
            typer.echo(note, err=True)
        raise
    result = recorded.result
    location: dict[str, object] = {"manifest": str(recorded.run_dir / "manifest.json")}
    if result.benchmark is not None:
        benchmark = result.benchmark
        return text_fields(
            {
                **location,
                "mode": result.mode.value,
                "status": "completed",
                "warmup": "completed",
                "steps": benchmark.total_steps,
                "chunks_executed": benchmark.chunks_executed,
                "initial_step": benchmark.initial_step,
                "final_step": benchmark.final_step,
                "elapsed_seconds": benchmark.elapsed_seconds,
                "steps_per_second": benchmark.steps_per_second,
                "platform": benchmark.platform,
                "device_kind": benchmark.device_kind,
            }
        )
    latest = consumer.latest_metrics
    if latest is None:
        raise RuntimeError("runtime completed without metrics")
    summary: dict[str, object] = {
        **location,
        "status": "completed",
        "steps_executed": result.runtime.steps_executed,
        "final_step": result.runtime.final_step,
        "alive_count": latest.alive_count,
        "births_total": consumer.births_total,
        "deaths_total": consumer.deaths_total,
        "active_lineage_count": latest.active_lineage_count,
        "final_total_energy": latest.total_energy_alive,
        "final_total_resources": latest.total_resources,
        "population_extinct": latest.alive_count == 0,
    }
    if result.smoke_report is not None:
        summary.update(mode=result.mode.value, initial_invariants="ok", final_invariants="ok")
    if result.latest_diagnostic is not None:
        summary["mode"] = result.mode.value
    return "" if quiet else text_fields(summary)
