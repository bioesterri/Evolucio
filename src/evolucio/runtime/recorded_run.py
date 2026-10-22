"""Recording adapter around fresh initialization and the existing mode dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from evolucio.config import ExperimentConfig
from evolucio.config.versions import CONFIG_SCHEMA_VERSION
from evolucio.host.schema import HOST_DTO_SCHEMA_DIGEST
from evolucio.persistence.local import (
    DEFAULT_RUNS_ROOT,
    LocalRunRecorder,
    RunErrorRecord,
    RunManifest,
    RunSchemas,
    RunStatus,
    create_local_run,
)
from evolucio.persistence.parquet.schemas import DEFAULT_PARQUET_POLICY, ParquetPersistencePolicy
from evolucio.persistence.zarr.schemas import DEFAULT_ZARR_POLICY, ZarrCapturePolicy

from .modes import ModeRunResult, RunDiagnosticError, RunMode
from .types import ChunkConsumer, HostChunkBatch, RunPlan

if TYPE_CHECKING:
    from evolucio.config.compile import CompiledConfig, CoreConfig
    from evolucio.persistence.zarr.checkpoint import CheckpointRestore

    from .capture_runner import RunCaptures


@dataclass(frozen=True, slots=True)
class RunSummary:
    steps_executed: int
    final_step: int
    alive_count: int
    births_total: int
    deaths_total: int
    active_lineage_count: int
    final_total_energy: float
    final_total_resources: float


def summarize_run(result: ModeRunResult) -> RunSummary:
    """Read only final population observations and accumulated runtime totals."""
    import jax
    import jax.numpy as jnp

    runtime = result.runtime
    population = runtime.state.population
    alive, lineages, births, deaths, energy, resources = jax.device_get(
        (
            population.alive,
            population.lineage_id,
            runtime.accumulator.births_total,
            runtime.accumulator.deaths_total,
            jnp.sum(jnp.where(population.alive, population.energy, 0)),
            jnp.sum(runtime.state.world.resources),
        )
    )
    return RunSummary(
        runtime.steps_executed,
        runtime.final_step,
        int(alive.sum()),
        int(births),
        int(deaths),
        len({int(value) for value in lineages[alive]}),
        float(energy),
        float(resources),
    )


@dataclass(frozen=True, slots=True)
class RecordedRunResult:
    result: ModeRunResult
    summary: RunSummary
    manifest: RunManifest
    run_dir: Path


def execute_recorded_simulation(
    *,
    recorder: LocalRunRecorder,
    capture_policy: ZarrCapturePolicy = DEFAULT_ZARR_POLICY,
    restored: CheckpointRestore | None = None,
    parquet_policy: ParquetPersistencePolicy = DEFAULT_PARQUET_POLICY,
    config: CoreConfig,
    plan: RunPlan,
    mode: RunMode,
    consumer: ChunkConsumer | None = None,
    consumer_factory: Callable[[], ChunkConsumer | None] | None = None,
    on_created: Callable[[LocalRunRecorder], None] | None = None,
) -> RecordedRunResult:
    """Keep initialization and execution failures auditable, preserving exceptions."""
    from evolucio.core.metrics import create_empty_metrics_accumulator

    from . import run_with_mode
    from .initialization import initialize_simulation

    if (
        recorder.manifest.status != RunStatus.CREATED
        or recorder.manifest.total_steps != plan.total_steps
        or recorder.manifest.chunk_steps != plan.chunk_steps
        or recorder.manifest.mode != mode
    ):
        raise ValueError("recorder and runtime plan/mode do not match")
    from evolucio.persistence.parquet import ParquetRunConsumer

    from .dto_conversion import DTOChunkConsumer

    parquet: ParquetRunConsumer | None = None
    captures: RunCaptures | None = None
    try:
        parquet = ParquetRunConsumer(recorder, parquet_policy)
        if capture_policy.enabled:
            from .capture_runner import RunCaptures

            captures = RunCaptures(recorder, config, capture_policy)
        if restored is not None:
            recorder.write_json_artifact(
                relative_path="artifacts/checkpoint_source.json",
                name="checkpoint_source",
                kind="checkpoint_provenance",
                payload={
                    "source_run_id": restored.run_id,
                    "source_checkpoint_step": restored.state_step,
                    "source_checkpoint_hash": restored.checkpoint_sha256,
                },
            )
        if on_created is not None:
            on_created(recorder)
        if restored is None:
            state = initialize_simulation(config, seed=recorder.manifest.seed)
            accumulator = create_empty_metrics_accumulator()
        else:
            if (
                restored.config_hash != recorder.manifest.config_hash
                or restored.compile_signature != recorder.manifest.compile_signature
                or restored.state_step != recorder.manifest.initial_step
            ):
                raise ValueError("restored carry does not match the recorded continuation")
            state, accumulator = restored.state, restored.accumulator
        sink = consumer if consumer_factory is None else consumer_factory()
        parquet_sink = DTOChunkConsumer(parquet)
        original_sink = sink

        def composed(batch: HostChunkBatch) -> None:
            parquet_sink(batch)
            if original_sink is not None:
                original_sink(batch)

        recorder.mark_running()
        if captures is None:
            outcome = run_with_mode(
                mode=mode,
                state=state,
                accumulator=accumulator,
                config=config,
                plan=plan,
                consumer=composed,
            )
        else:
            from .capture_runner import run_with_captures

            outcome = run_with_captures(
                mode=mode,
                state=state,
                accumulator=accumulator,
                config=config,
                plan=plan,
                consumer=composed,
                captures=captures,
            )
        summary = summarize_run(outcome)
        if captures is not None:
            captures.writer.finalize()
        parquet_summary = parquet.finalize()
        recorder.mark_completed(
            summary=asdict(summary) | asdict(parquet_summary),
            steps_executed=summary.steps_executed,
            final_step=summary.final_step,
        )
        return RecordedRunResult(outcome, summary, recorder.manifest, recorder.run_dir)
    except Exception as exc:
        if captures is not None and not captures.writer.closed:
            try:
                captures.writer.finalize(complete=False)
            except Exception as capture_error:
                exc.add_note(f"Could not finalize partial Zarr index: {capture_error}")
        if parquet is not None and not parquet.closed:
            try:
                parquet.finalize_partial()
            except Exception as partial_error:
                exc.add_note(f"Could not finalize partial Parquet: {partial_error}")
        try:
            recorder.mark_failed(
                RunErrorRecord(
                    type(exc).__name__,
                    str(exc)[:4096],
                    exc.failure.step if isinstance(exc, RunDiagnosticError) else None,
                )
            )
        except Exception as recording_error:
            exc.add_note(
                f"Could not record failure at {recorder.run_dir / 'manifest.json'}: "
                f"{type(recording_error).__name__}: {str(recording_error)[:512]}"
            )
        raise


def run_recorded_simulation(
    *,
    config: ExperimentConfig,
    capture_policy: ZarrCapturePolicy = DEFAULT_ZARR_POLICY,
    source_checkpoint: Path | None = None,
    parquet_policy: ParquetPersistencePolicy = DEFAULT_PARQUET_POLICY,
    plan: RunPlan,
    mode: RunMode = RunMode.EXPERIMENT,
    output_root: Path = DEFAULT_RUNS_ROOT,
    consumer: ChunkConsumer | None = None,
    run_id: str | None = None,
    on_created: Callable[[LocalRunRecorder], None] | None = None,
    compiled: CompiledConfig | None = None,
    case_id: str | None = None,
    batch_ordinal: int | None = None,
) -> RecordedRunResult:
    """Require the canonical host model so the frozen config is exactly executable."""
    from evolucio.config.compile import compile_config, compile_signature_digest
    from evolucio.config.freeze import freeze_config

    prepared = compile_config(config) if compiled is None else compiled
    if prepared.config_hash != freeze_config(config).config_hash:
        raise ValueError("compiled configuration does not match the frozen host configuration")
    restored = None
    if source_checkpoint is not None:
        from evolucio.persistence.zarr.checkpoint import load_checkpoint

        restored = load_checkpoint(
            checkpoint_path=source_checkpoint,
            expected_config_hash=prepared.config_hash,
            expected_compile_signature=compile_signature_digest(prepared.compile_signature),
        )
    recorder = create_local_run(
        config=config,
        plan=plan,
        mode=mode,
        output_root=output_root,
        run_id=run_id,
        compile_signature=compile_signature_digest(prepared.compile_signature),
        schemas=RunSchemas(
            CONFIG_SCHEMA_VERSION,
            HOST_DTO_SCHEMA_DIGEST,
            prepared.compile_signature.simulation_step_schema_digest,
            prepared.compile_signature.metrics_schema_digest,
        ),
        initial_step=0 if restored is None else restored.state_step,
        case_id=case_id,
        batch_ordinal=batch_ordinal,
    )
    return execute_recorded_simulation(
        recorder=recorder,
        capture_policy=capture_policy,
        restored=restored,
        parquet_policy=parquet_policy,
        config=prepared.core,
        plan=plan,
        mode=mode,
        consumer=consumer,
        on_created=on_created,
    )
