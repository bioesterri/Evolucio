"""Recording adapter around fresh initialization and the existing mode dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from evolucio.config import ExperimentConfig
from evolucio.config.versions import CONFIG_SCHEMA_VERSION
from evolucio.host import ChunkTelemetryDTO
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
from evolucio.persistence.local.manifest import RunStopRecord, utc_now
from evolucio.persistence.local.ownership import run_ownership
from evolucio.persistence.parquet.schemas import DEFAULT_PARQUET_POLICY, ParquetPersistencePolicy
from evolucio.persistence.zarr.schemas import DEFAULT_ZARR_POLICY, ZarrCapturePolicy

from .modes import ModeRunResult, RunDiagnosticError, RunMode
from .stop_control import StopController
from .types import ChunkConsumer, HostChunkBatch, RunPlan

if TYPE_CHECKING:
    from evolucio.config.compile import CompiledConfig, CoreConfig
    from evolucio.persistence.parquet import ParquetRunConsumer
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


class RecordedRunObserver(Protocol):
    """Host lifecycle hooks for optional registry adapters."""

    def created(self, recorder: LocalRunRecorder) -> None: ...
    def running(self, recorder: LocalRunRecorder) -> None: ...
    def completing(
        self, recorder: LocalRunRecorder, manifest: RunManifest, summary: RunSummary
    ) -> None: ...
    def failed(self, recorder: LocalRunRecorder, error: Exception) -> None: ...
    def stopped(self, recorder: LocalRunRecorder) -> None: ...


class RecordedRunTracking(Protocol):
    """Derived tracking receives the canonical DTOs and terminal manifest."""

    def created(self, recorder: LocalRunRecorder) -> None: ...
    def consume(self, recorder: LocalRunRecorder, telemetry: ChunkTelemetryDTO) -> None: ...
    def finished(self, recorder: LocalRunRecorder) -> None: ...


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


def _execute_recorded_simulation(
    *,
    recorder: LocalRunRecorder,
    capture_policy: ZarrCapturePolicy = DEFAULT_ZARR_POLICY,
    restored: CheckpointRestore | None = None,
    checkpoint_source_run: Path | None = None,
    parquet_policy: ParquetPersistencePolicy = DEFAULT_PARQUET_POLICY,
    config: CoreConfig,
    plan: RunPlan,
    mode: RunMode,
    consumer: ChunkConsumer | None = None,
    consumer_factory: Callable[[], ChunkConsumer | None] | None = None,
    on_created: Callable[[LocalRunRecorder], None] | None = None,
    observer: RecordedRunObserver | None = None,
    tracking: RecordedRunTracking | None = None,
    telemetry_consumer: Callable[[ChunkTelemetryDTO], None] | None = None,
    stop_controller: StopController,
    import_prefix: Callable[[ParquetRunConsumer], None] | None = None,
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
    from evolucio.analysis.summaries import FOUNDERS_PATH
    from evolucio.persistence.parquet import ParquetRunConsumer

    from .analysis_capture import AnalysisCapture
    from .dto_conversion import DTOChunkConsumer

    parquet: ParquetRunConsumer | None = None
    captures: RunCaptures | None = None
    analysis: AnalysisCapture | None = None
    tracking_preflight = False
    stopped = False
    phase = "initialization"
    confirmed: int | None = None
    active_interval: tuple[int, int] | None = None

    def stop(reason: str, requested_at: str | None, *, ordered: bool) -> None:
        recorder.mark_stopped(
            RunStopRecord(
                reason,
                requested_at,
                utc_now(),
                ordered,
                confirmed,
                None if parquet is None else parquet.last_end,
                None
                if captures is None or not captures.writer.index.checkpoints
                else captures.writer.index.checkpoints[-1].state_step,
            )
        )

    try:
        if on_created is not None:
            on_created(recorder)
        if observer is not None:
            phase = "database"
            observer.created(recorder)
        phase = "initialization"
        if tracking is not None:
            tracking_preflight = True
            tracking.created(recorder)
            tracking_preflight = False
        phase = "persistence"
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
        phase = "initialization"
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
        confirmed = int(state.step)
        phase = "persistence"
        if import_prefix is not None:
            if restored is None or checkpoint_source_run is None:
                raise ValueError("prefix import requires a restored checkpoint")
            import_prefix(parquet)
            analysis = AnalysisCapture.import_founders(recorder, checkpoint_source_run)
            if captures is None:
                from .capture_runner import RunCaptures

                captures = RunCaptures(recorder, config, capture_policy)
            captures.checkpoint(state, accumulator)
        elif restored is None or restored.state_step == 0:
            analysis = AnalysisCapture.start(recorder, state)
        elif (
            checkpoint_source_run is not None and (checkpoint_source_run / FOUNDERS_PATH).is_file()
        ):
            analysis = AnalysisCapture.resume(
                recorder,
                checkpoint_source_run,
                restored.run_id,
                restored.state_step,
            )
        sink = consumer if consumer_factory is None else consumer_factory()
        parquet_consumer = parquet

        def persist(telemetry: ChunkTelemetryDTO) -> None:
            parquet_consumer(telemetry)
            if tracking is not None:
                tracking.consume(recorder, telemetry)
            if mode is not RunMode.BENCHMARK and telemetry_consumer is not None:
                telemetry_consumer(telemetry)

        parquet_sink = DTOChunkConsumer(persist)

        def composed(batch: HostChunkBatch) -> None:
            nonlocal phase, confirmed, active_interval
            confirmed = batch.end_step_exclusive
            active_interval = (batch.start_step, batch.end_step_exclusive)
            phase = "persistence"
            parquet_sink(batch)
            if mode is not RunMode.BENCHMARK and sink is not None:
                phase = "unexpected"
                sink(batch)
            phase = "simulation"
            active_interval = None

        def dispatch(start: int, end: int) -> None:
            nonlocal phase, active_interval
            active_interval = (start, end)
            phase = "simulation"

        recorder.mark_running()
        if observer is not None:
            phase = "database"
            observer.running(recorder)
        phase = "simulation"
        if captures is None:
            outcome = run_with_mode(
                mode=mode,
                state=state,
                accumulator=accumulator,
                config=config,
                plan=plan,
                consumer=composed,
                benchmark_consumer=composed,
                stop_controller=stop_controller,
                on_dispatch=dispatch,
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
                stop_controller=stop_controller,
                on_dispatch=dispatch,
            )
        phase = "persistence"
        if outcome.runtime.steps_executed < plan.total_steps:
            request = stop_controller.request
            assert request is not None
            if capture_policy.checkpoint_on_stop and (
                parquet.last_end == outcome.runtime.final_step or outcome.runtime.final_step == 0
            ):
                if captures is None:
                    from .capture_runner import RunCaptures

                    captures = RunCaptures(recorder, config, capture_policy)
                captures.checkpoint(outcome.runtime.state, outcome.runtime.accumulator)
            if captures is not None:
                captures.writer.finalize(complete=False)
            parquet.finalize_partial()
            if analysis is not None:
                analysis.register_index()
            stop(*request, ordered=True)
            stopped = True
            if observer is not None and hasattr(observer, "stopped"):
                observer.stopped(recorder)
            if tracking is not None:
                tracking.finished(recorder)
            return RecordedRunResult(
                outcome, summarize_run(outcome), recorder.manifest, recorder.run_dir
            )
        if captures is not None:
            captures.writer.finalize()
        parquet_summary = parquet.finalize()
        summary = summarize_run(outcome)
        analysis_summary: dict[str, object] = (
            {"analysis_status": "unavailable_pre_checkpoint_history"}
            if analysis is None
            else analysis.finalize(
                outcome.runtime.state,
                summary.final_step,
                births_total=summary.births_total,
                deaths_total=summary.deaths_total,
            )
        )

        def publish_completion(manifest: RunManifest) -> None:
            nonlocal phase
            if observer is not None:
                phase = "database"
                observer.completing(recorder, manifest, summary)
                phase = "persistence"

        benchmark_summary: dict[str, object] = {}
        if outcome.benchmark is not None:
            benchmark_summary = {
                "elapsed_seconds": outcome.benchmark.elapsed_seconds,
                "steps_per_second": outcome.benchmark.steps_per_second,
            }
        trajectory_summary: dict[str, object] = {}
        if import_prefix is not None:
            trajectory_summary = {
                "trajectory_initial_step": 0,
                "execution_initial_step": recorder.manifest.initial_step,
                "target_step": recorder.manifest.initial_step + plan.total_steps,
                "inherited_steps": recorder.manifest.initial_step,
                "new_steps_executed": summary.steps_executed,
                "trajectory_steps_completed": summary.final_step,
            }
        recorder.mark_completed(
            summary=asdict(summary)
            | asdict(parquet_summary)
            | analysis_summary
            | benchmark_summary
            | trajectory_summary,
            steps_executed=summary.steps_executed,
            final_step=summary.final_step,
            before_publish=None if observer is None else publish_completion,
        )
    except (Exception, KeyboardInterrupt) as exc:
        if (tracking_preflight and isinstance(exc, Exception)) or stopped:
            raise
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
        if analysis is not None and not any(
            artifact.kind == "highlighted_genome_index" for artifact in recorder.manifest.artifacts
        ):
            try:
                analysis.register_index()
            except Exception as index_error:
                exc.add_note(f"Could not register the genome index: {index_error}")
        try:
            if isinstance(exc, KeyboardInterrupt):
                stop("user_interruption", None, ordered=False)
            else:
                recorder.mark_failed(
                    RunErrorRecord(
                        type(exc).__name__,
                        str(exc)[:4096],
                        exc.failure.step if isinstance(exc, RunDiagnosticError) else None,
                        "invariant_violation"
                        if isinstance(exc, RunDiagnosticError)
                        else f"{phase}_error",
                        phase,
                        confirmed,
                        None if parquet is None else parquet.last_end,
                        None
                        if captures is None or not captures.writer.index.checkpoints
                        else captures.writer.index.checkpoints[-1].state_step,
                        utc_now(),
                        active_interval,
                    )
                )
        except Exception as recording_error:
            exc.add_note(
                f"Could not record failure at {recorder.run_dir / 'manifest.json'}: "
                f"{type(recording_error).__name__}: {str(recording_error)[:512]}"
            )
        if observer is not None:
            try:
                if isinstance(exc, Exception):
                    observer.failed(recorder, exc)
                elif hasattr(observer, "stopped"):
                    observer.stopped(recorder)
            except Exception as registry_error:
                exc.add_note(f"Could not record registry failure: {registry_error}")
        if tracking is not None:
            try:
                tracking.finished(recorder)
            except Exception:
                exc.add_note("Could not finalize MLflow tracking for the failed run")
        raise
    if tracking is not None:
        tracking.finished(recorder)
    return RecordedRunResult(outcome, summary, recorder.manifest, recorder.run_dir)


def execute_recorded_simulation(
    *,
    recorder: LocalRunRecorder,
    capture_policy: ZarrCapturePolicy = DEFAULT_ZARR_POLICY,
    restored: CheckpointRestore | None = None,
    checkpoint_source_run: Path | None = None,
    parquet_policy: ParquetPersistencePolicy = DEFAULT_PARQUET_POLICY,
    config: CoreConfig,
    plan: RunPlan,
    mode: RunMode,
    consumer: ChunkConsumer | None = None,
    consumer_factory: Callable[[], ChunkConsumer | None] | None = None,
    on_created: Callable[[LocalRunRecorder], None] | None = None,
    observer: RecordedRunObserver | None = None,
    tracking: RecordedRunTracking | None = None,
    telemetry_consumer: Callable[[ChunkTelemetryDTO], None] | None = None,
    stop_controller: StopController | None = None,
    import_prefix: Callable[[ParquetRunConsumer], None] | None = None,
) -> RecordedRunResult:
    stop_controller = stop_controller or StopController()
    with run_ownership(recorder.run_dir, create=True) as ownership, stop_controller.handle_sigint():
        if ownership != "inactive":
            raise ValueError(f"run ownership unavailable: {ownership}")
        return _execute_recorded_simulation(
            recorder=recorder,
            capture_policy=capture_policy,
            restored=restored,
            checkpoint_source_run=checkpoint_source_run,
            parquet_policy=parquet_policy,
            config=config,
            plan=plan,
            mode=mode,
            consumer=consumer,
            consumer_factory=consumer_factory,
            on_created=on_created,
            observer=observer,
            tracking=tracking,
            telemetry_consumer=telemetry_consumer,
            import_prefix=import_prefix,
            stop_controller=stop_controller,
        )


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
    observer: RecordedRunObserver | None = None,
    tracking: RecordedRunTracking | None = None,
    telemetry_consumer: Callable[[ChunkTelemetryDTO], None] | None = None,
    compiled: CompiledConfig | None = None,
    case_id: str | None = None,
    batch_ordinal: int | None = None,
    stop_controller: StopController | None = None,
) -> RecordedRunResult:
    """Require the canonical host model so the frozen config is exactly executable."""
    import jax

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
        backend=jax.default_backend(),
    )
    return execute_recorded_simulation(
        recorder=recorder,
        capture_policy=capture_policy,
        restored=restored,
        checkpoint_source_run=(
            None
            if source_checkpoint is None
            else next(
                (
                    parent
                    for parent in source_checkpoint.resolve().parents
                    if (parent / "manifest.json").is_file()
                ),
                None,
            )
        ),
        parquet_policy=parquet_policy,
        config=prepared.core,
        plan=plan,
        mode=mode,
        consumer=consumer,
        on_created=on_created,
        observer=observer,
        tracking=tracking,
        telemetry_consumer=telemetry_consumer,
        stop_controller=stop_controller,
    )
