"""Bridge recorded host runs to the optional PostgreSQL registry."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from uuid import UUID

from evolucio.analysis.summaries import AGENTS_PATH
from evolucio.persistence.local import LocalRunRecorder, RunManifest
from evolucio.recovery.provenance import read_provenance
from evolucio.runtime.recorded_run import RunSummary

from .errors import RegistryIntegrityError
from .reconcile import verified_run_indexes
from .registry import create_registry_engine, create_session_factory
from .repository import PostgresRegistry
from .types import RunSummaryRecord


class PostgresRunObserver:
    def __init__(self, registry: PostgresRegistry, experiment_id: UUID) -> None:
        self.registry = registry
        self.experiment_id = experiment_id
        self.registered_run_id: UUID | None = None

    def created(self, recorder: LocalRunRecorder) -> None:
        recorder.write_json_artifact(
            relative_path="artifacts/postgres/experiment.json",
            name="postgres_experiment",
            kind="postgres_experiment",
            payload={"experiment_id": str(self.experiment_id)},
        )
        manifest = recorder.manifest
        artifact = next(
            a for a in manifest.artifacts if a.relative_path == manifest.config_artifact
        )
        self.registry.create_run(
            manifest=manifest,
            config_json=json.loads(
                (recorder.run_dir / artifact.relative_path).read_text(encoding="utf-8")
            ),
            config_schema_version=int(manifest.schemas.config.split(".", maxsplit=1)[0]),
            config_artifact_sha256=artifact.sha256,
            experiment_id=self.experiment_id,
            local_run_path=str(recorder.run_dir),
            recovery=read_provenance(recorder.run_dir, manifest),
        )
        self.registered_run_id = UUID(recorder.run_id)

    def running(self, recorder: LocalRunRecorder) -> None:
        started = recorder.manifest.started_at_utc
        assert started is not None
        self.registry.mark_run_running(
            UUID(recorder.run_id), started_at=datetime.fromisoformat(started)
        )

    def completing(
        self, recorder: LocalRunRecorder, manifest: RunManifest, summary: RunSummary
    ) -> None:
        _, indexes = verified_run_indexes(recorder.run_dir)
        assert manifest.finished_at_utc is not None
        self.registry.mark_run_completed(
            UUID(manifest.run_id),
            summary=RunSummaryRecord(
                summary.alive_count,
                summary.births_total,
                summary.deaths_total,
                summary.active_lineage_count,
                summary.final_total_energy,
                summary.final_total_resources,
            ),
            final_step=summary.final_step,
            steps_executed=summary.steps_executed,
            finished_at=datetime.fromisoformat(manifest.finished_at_utc),
            indexes=indexes,
            analysis_run_dir=(
                recorder.run_dir
                if any(artifact.relative_path == AGENTS_PATH for artifact in indexes.artifacts)
                else None
            ),
        )

    def stopped(self, recorder: LocalRunRecorder) -> None:
        if self.registered_run_id != UUID(recorder.run_id):
            return
        self.registry.mark_run_stopped(recorder.manifest)
        _, indexes = verified_run_indexes(recorder.run_dir)
        self.registry.mark_run_stopped(recorder.manifest, indexes=indexes)

    def failed(self, recorder: LocalRunRecorder, error: Exception) -> None:
        run_id = UUID(recorder.run_id)
        if self.registered_run_id != run_id:
            return
        manifest = recorder.manifest
        details = manifest.error
        finished = manifest.finished_at_utc
        error_type = details.error_type if details is not None else type(error).__name__
        error_message = details.message if details is not None else str(error)[:4096]
        error_step = details.step if details is not None else None
        finished_at = datetime.fromisoformat(finished) if finished is not None else None
        try:
            _, indexes = verified_run_indexes(recorder.run_dir)
            self.registry.mark_run_failed(
                run_id,
                error_type=error_type,
                error_message=error_message,
                error_step=error_step,
                finished_at=finished_at,
                indexes=indexes,
            )
        except RegistryIntegrityError as exc:
            error.add_note(f"Could not index partial artifacts: {exc}")
            self.registry.mark_run_failed(
                run_id,
                error_type=error_type,
                error_message=error_message,
                error_step=error_step,
                finished_at=finished_at,
            )


@contextmanager
def register_experiment_runs(experiment_id: UUID) -> Generator[PostgresRunObserver]:
    engine = create_registry_engine()
    try:
        registry = PostgresRegistry(create_session_factory(engine))
        if registry.get_experiment(experiment_id) is None:
            raise RegistryIntegrityError(f"unknown experiment_id: {experiment_id}")
        yield PostgresRunObserver(registry, experiment_id)
    finally:
        engine.dispose()
