"""Orchestrate a new recorded run from a verified checkpoint."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from evolucio.config.compile import compile_config
from evolucio.persistence.local import LocalRunRecorder, RunStatus, create_local_run
from evolucio.persistence.local.manifest import RunStopRecord, load_run_manifest, utc_now
from evolucio.persistence.local.ownership import run_ownership
from evolucio.persistence.parquet.index import load_index as load_telemetry
from evolucio.persistence.parquet.index import prefix_digest
from evolucio.persistence.zarr.index import load_index
from evolucio.runtime.recorded_run import RecordedRunResult, execute_recorded_simulation
from evolucio.runtime.types import RunPlan

from .assessment import RecoveryAssessment, inspect_owned, load_canonical_config, restore_indexed
from .errors import (
    CheckpointNotRecoverableError,
    RecoveryPersistenceError,
    RecoveryPlanError,
    RunStillActiveError,
)
from .history import import_history, verify_history
from .plan import RecoveryPlan
from .provenance import PROVENANCE_PATH, RecoveryProvenance, read_provenance

if TYPE_CHECKING:
    from evolucio.persistence.parquet.writer import ParquetRunConsumer
    from evolucio.runtime.recorded_run import RecordedRunObserver, RecordedRunTracking
    from evolucio.runtime.stop_control import StopController


def resume(
    run_dir: Path,
    *,
    checkpoint_step: int | None = None,
    chunk_steps: int | None = None,
    dry_run: bool = False,
    show_plan: Callable[[RecoveryPlan, RecoveryAssessment], None] | None = None,
    stop_controller: StopController | None = None,
    observer: RecordedRunObserver | None = None,
    tracking: RecordedRunTracking | None = None,
) -> RecoveryPlan | RecordedRunResult:
    with run_ownership(run_dir) as ownership:
        if ownership == "active":
            raise RunStillActiveError("source writer is active")
        assessment = inspect_owned(run_dir, ownership)
        step = (
            assessment.latest_recoverable_checkpoint_step
            if checkpoint_step is None
            else checkpoint_step
        )
        if step is None or step not in assessment.recoverable_steps:
            details = (*assessment.incompatibilities, *assessment.integrity_errors)
            raise CheckpointNotRecoverableError(f"checkpoint {step} is not recoverable: {details}")
        manifest = load_run_manifest(run_dir / "manifest.json")
        if manifest.status is RunStatus.RUNNING and not dry_run:
            raise RecoveryPlanError("reconcile the interrupted source before resuming")
        zarr_index = load_index(run_dir)
        record = next(record for record in zarr_index.checkpoints if record.state_step == step)
        plan = RecoveryPlan(
            manifest.run_id,
            step,
            record.sha256,
            assessment.target_step,
            assessment.target_step - step,
            manifest.config_hash,
            manifest.chunk_steps if chunk_steps is None else chunk_steps,
        )
        if show_plan is not None:
            show_plan(plan, assessment)
        if dry_run:
            return plan
        if ownership != "inactive":
            raise RunStillActiveError("exclusive source ownership cannot be established")
        previous = read_provenance(run_dir, manifest)
        if previous is None and manifest.initial_step != 0:
            raise RecoveryPlanError("the original trajectory is unknown")
        index = load_telemetry(run_dir)
        restored = restore_indexed(run_dir, manifest, record)
        counts = verify_history(run_dir, manifest, index, restored)
        provenance = RecoveryProvenance(
            1,
            manifest.run_id,
            manifest.run_id if previous is None else previous.root_run_id,
            1 if previous is None else previous.recovery_depth + 1,
            step,
            record.sha256,
            manifest.config_hash,
            step,
            plan.remaining_steps,
            plan.target_step,
            counts["metrics"],
            counts["births"],
            counts["deaths"],
            counts["genealogy"],
            prefix_digest(index, step),
        )
        config = load_canonical_config(run_dir, manifest)
        compiled = compile_config(config)
        runtime_plan = RunPlan(plan.remaining_steps, plan.chunk_steps)
        recorder = create_local_run(
            config=config,
            plan=runtime_plan,
            mode=manifest.mode,
            output_root=run_dir.parent,
            compile_signature=manifest.compile_signature,
            schemas=manifest.schemas,
            initial_step=step,
            case_id=manifest.case_id,
            batch_ordinal=manifest.batch_ordinal,
            backend=manifest.environment.backend,
        )

        def write_provenance(value: LocalRunRecorder) -> None:
            value.write_json_artifact(
                relative_path=PROVENANCE_PATH,
                payload=asdict(provenance),
                name="recovery_provenance",
                kind="recovery_provenance",
            )

        def materialize(writer: ParquetRunConsumer) -> None:
            try:
                import_history(run_dir, index, step, writer)
            except (OSError, ValueError) as exc:
                raise RecoveryPersistenceError("could not commit inherited telemetry") from exc

        return execute_recorded_simulation(
            recorder=recorder,
            config=compiled.core,
            plan=runtime_plan,
            mode=manifest.mode,
            restored=restored,
            checkpoint_source_run=run_dir,
            capture_policy=replace(zarr_index.policy, keep_last_checkpoints=None),
            parquet_policy=index.policy,
            on_created=write_provenance,
            import_prefix=materialize,
            stop_controller=stop_controller,
            observer=observer,
            tracking=tracking,
        )


def reconcile(run_dir: Path) -> RecoveryAssessment:
    with run_ownership(run_dir) as ownership:
        if ownership != "inactive":
            raise RunStillActiveError(f"exclusive source ownership unavailable: {ownership}")
        recorder = LocalRunRecorder.open_existing(run_dir)
        assessment = inspect_owned(run_dir, ownership)
        if recorder.manifest.status is RunStatus.RUNNING:
            recorder.mark_stopped(
                RunStopRecord(
                    "external_interruption",
                    None,
                    utc_now(),
                    False,
                    None,
                    assessment.last_committed_telemetry_step,
                    assessment.latest_valid_checkpoint_step,
                )
            )
            assessment = inspect_owned(run_dir, ownership)
        return assessment
