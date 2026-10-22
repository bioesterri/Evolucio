"""Transactional repository for experiments, runs and artifact indexes."""

from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from evolucio.config.freeze import canonical_json_and_hash
from evolucio.persistence.local.artifacts import ArtifactRecord, validate_sha256
from evolucio.persistence.local.manifest import RunManifest, RunStatus

from .errors import (
    ArtifactIndexConflictError,
    InvalidRunTransitionError,
    RegistryIntegrityError,
    RunAlreadyExistsError,
)
from .models import (
    ArtifactModel,
    ConfigSnapshotModel,
    DatasetIndexModel,
    ExperimentModel,
    RunModel,
    RunSummaryModel,
    StateCaptureModel,
)
from .types import ComparisonRun, ExperimentRecord, RunSummaryRecord


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value.replace("Z", "+00:00"))


class ExperimentRepository(Protocol):
    def create_experiment(self, *, name: str, **metadata: str | None) -> ExperimentRecord: ...
    def get_experiment(self, experiment_id: UUID) -> ExperimentRecord | None: ...
    def list_experiments(self) -> tuple[ExperimentRecord, ...]: ...


class RunRepository(Protocol):
    def get_run(self, run_id: UUID) -> RunModel | None: ...
    def list_runs(self, **filters: object) -> tuple[RunModel, ...]: ...


class PostgresRegistry:
    """Short-transaction façade; sessions never span simulation execution."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sessions = session_factory

    def create_experiment(
        self,
        *,
        name: str,
        description: str | None = None,
        objective: str | None = None,
        hypothesis: str | None = None,
        label: str | None = None,
        status: str = "created",
        experiment_id: UUID | None = None,
    ) -> ExperimentRecord:
        if not name.strip() or not status.strip():
            raise ValueError("experiment name and status must be nonempty")
        now = _now()
        model = ExperimentModel(
            experiment_id=experiment_id or uuid4(),
            name=name,
            description=description,
            objective=objective,
            hypothesis=hypothesis,
            label=label,
            status=status,
            created_at=now,
            updated_at=now,
        )
        try:
            with self._sessions.begin() as session:
                session.add(model)
        except IntegrityError as exc:
            raise RegistryIntegrityError("experiment violates relational constraints") from exc
        return self._experiment_record(model)

    def get_experiment(self, experiment_id: UUID) -> ExperimentRecord | None:
        with self._sessions() as session:
            model = session.get(ExperimentModel, experiment_id)
            return None if model is None else self._experiment_record(model)

    def list_experiments(self) -> tuple[ExperimentRecord, ...]:
        with self._sessions() as session:
            models = session.scalars(select(ExperimentModel).order_by(ExperimentModel.created_at))
            return tuple(self._experiment_record(model) for model in models)

    def create_run(
        self,
        *,
        manifest: RunManifest,
        config_json: dict[str, object],
        config_schema_version: int,
        config_artifact_sha256: str,
        experiment_id: UUID | None = None,
        local_run_path: str | None = None,
    ) -> RunModel:
        """Atomically create the PR-35 run identity and immutable config snapshot."""
        if manifest.status is not RunStatus.CREATED:
            raise ValueError("only a created manifest can be registered")
        validate_sha256(config_artifact_sha256)
        _, snapshot_hash = canonical_json_and_hash(config_json)
        if snapshot_hash != manifest.config_hash:
            raise RegistryIntegrityError("config snapshot differs from run config_hash")
        run = self._run_from_manifest(manifest, experiment_id, local_run_path)
        snapshot = ConfigSnapshotModel(
            run_id=run.run_id,
            config_hash=run.config_hash,
            schema_version=config_schema_version,
            config_json=config_json,
            artifact_relative_path="config.json",
            artifact_sha256=config_artifact_sha256,
            created_at=_now(),
        )
        try:
            with self._sessions.begin() as session:
                if (
                    experiment_id is not None
                    and session.get(ExperimentModel, experiment_id) is None
                ):
                    raise RegistryIntegrityError("experiment does not exist")
                # There is deliberately no mutable ORM relationship between a run and
                # its immutable snapshot. Flush the parent explicitly so PostgreSQL
                # always observes the FK dependency while preserving one transaction.
                session.add(run)
                session.flush()
                session.add(snapshot)
        except IntegrityError as exc:
            with self._sessions() as session:
                if session.get(RunModel, run.run_id) is not None:
                    raise RunAlreadyExistsError(f"run {manifest.run_id} already exists") from exc
            raise RegistryIntegrityError("atomic run/config insertion failed") from exc
        return run

    def get_run(self, run_id: UUID) -> RunModel | None:
        with self._sessions() as session:
            return session.get(RunModel, run_id)

    def list_runs(
        self,
        *,
        experiment_id: UUID | None = None,
        config_hash: str | None = None,
        seed: int | None = None,
        status: str | None = None,
    ) -> tuple[RunModel, ...]:
        query: Select[tuple[RunModel]] = select(RunModel)
        for criterion in (
            RunModel.experiment_id == experiment_id if experiment_id is not None else None,
            RunModel.config_hash == config_hash if config_hash is not None else None,
            RunModel.seed == seed if seed is not None else None,
            RunModel.status == status if status is not None else None,
        ):
            if criterion is not None:
                query = query.where(criterion)
        with self._sessions() as session:
            return tuple(session.scalars(query.order_by(RunModel.created_at)))

    def mark_run_running(self, run_id: UUID, *, started_at: datetime | None = None) -> None:
        self._transition(run_id, "created", "running", started_at=started_at or _now())

    def mark_run_failed(
        self,
        run_id: UUID,
        *,
        error_type: str,
        error_message: str,
        error_step: int | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        if not error_type:
            raise ValueError("error_type must be nonempty")
        with self._sessions.begin() as session:
            run = self._required_run(session, run_id)
            if run.status not in {"created", "running"}:
                raise InvalidRunTransitionError(f"cannot fail run in status {run.status}")
            run.status, run.finished_at = "failed", finished_at or _now()
            run.error_type, run.error_message, run.error_step = (
                error_type,
                error_message,
                error_step,
            )

    def mark_run_completed(
        self,
        run_id: UUID,
        *,
        summary: RunSummaryRecord,
        final_step: int,
        steps_executed: int,
        finished_at: datetime | None = None,
    ) -> None:
        """Atomically install one immutable summary and close the run lifecycle."""
        with self._sessions.begin() as session:
            run = self._required_run(session, run_id)
            if run.status != "running":
                raise InvalidRunTransitionError(f"cannot complete run in status {run.status}")
            if session.get(RunSummaryModel, run_id) is not None:
                raise RegistryIntegrityError("run summary is immutable")
            artifact_paths = set(
                session.scalars(
                    select(ArtifactModel.relative_path).where(ArtifactModel.run_id == run_id)
                )
            )
            dataset_kinds = set(
                session.scalars(
                    select(DatasetIndexModel.dataset_kind).where(
                        DatasetIndexModel.run_id == run_id,
                        DatasetIndexModel.complete.is_(True),
                    )
                )
            )
            if not {"config.json", "summary.json"} <= artifact_paths or dataset_kinds != {
                "metrics",
                "births",
                "deaths",
                "genealogy",
            }:
                raise RegistryIntegrityError(
                    "completed run requires verified summary/config and complete dataset indexes"
                )
            session.add(
                RunSummaryModel(
                    run_id=run_id,
                    alive_count=summary.alive_count,
                    births_total=summary.births_total,
                    deaths_total=summary.deaths_total,
                    active_lineage_count=summary.active_lineage_count,
                    final_total_energy=summary.final_total_energy,
                    final_total_resources=summary.final_total_resources,
                    population_extinct=summary.population_extinct,
                    created_at=_now(),
                )
            )
            run.status, run.finished_at = "completed", finished_at or _now()
            run.final_step, run.steps_executed = final_step, steps_executed

    def index_artifact(self, run_id: UUID, artifact: ArtifactRecord) -> None:
        with self._sessions.begin() as session:
            existing = session.scalar(
                select(ArtifactModel).where(
                    ArtifactModel.run_id == run_id,
                    ArtifactModel.relative_path == artifact.relative_path,
                )
            )
            if existing is not None:
                if existing.sha256 != artifact.sha256 or existing.byte_size != artifact.byte_size:
                    raise ArtifactIndexConflictError(artifact.relative_path)
                return
            session.add(
                ArtifactModel(
                    artifact_id=uuid4(),
                    run_id=run_id,
                    name=artifact.name,
                    kind=artifact.kind,
                    relative_path=artifact.relative_path,
                    media_type=artifact.media_type,
                    byte_size=artifact.byte_size,
                    sha256=artifact.sha256,
                    schema_name=None,
                    schema_version=None,
                    created_at=_now(),
                )
            )

    def list_artifacts(self, run_id: UUID) -> tuple[ArtifactModel, ...]:
        with self._sessions() as session:
            return tuple(
                session.scalars(select(ArtifactModel).where(ArtifactModel.run_id == run_id))
            )

    def list_datasets(self, run_id: UUID) -> tuple[DatasetIndexModel, ...]:
        with self._sessions() as session:
            return tuple(
                session.scalars(select(DatasetIndexModel).where(DatasetIndexModel.run_id == run_id))
            )

    def list_state_captures(self, run_id: UUID) -> tuple[StateCaptureModel, ...]:
        with self._sessions() as session:
            return tuple(
                session.scalars(select(StateCaptureModel).where(StateCaptureModel.run_id == run_id))
            )

    def index_dataset(self, record: DatasetIndexModel) -> None:
        with self._sessions.begin() as session:
            existing = session.scalar(
                select(DatasetIndexModel).where(
                    DatasetIndexModel.run_id == record.run_id,
                    DatasetIndexModel.dataset_kind == record.dataset_kind,
                )
            )
            if existing is not None:
                if existing.index_sha256 != record.index_sha256 or any(
                    getattr(existing, field) != getattr(record, field)
                    for field in (
                        "relative_path",
                        "row_count",
                        "part_count",
                        "min_step",
                        "max_step",
                        "complete",
                    )
                ):
                    raise ArtifactIndexConflictError(f"dataset:{record.dataset_kind}")
                return
            session.add(record)

    def index_state_capture(self, record: StateCaptureModel) -> None:
        with self._sessions.begin() as session:
            existing = session.scalar(
                select(StateCaptureModel).where(
                    StateCaptureModel.run_id == record.run_id,
                    StateCaptureModel.capture_kind == record.capture_kind,
                    StateCaptureModel.state_step == record.state_step,
                )
            )
            if existing is not None:
                if (
                    existing.sha256 != record.sha256
                    or existing.relative_path != record.relative_path
                ):
                    raise ArtifactIndexConflictError(record.relative_path)
                return
            session.add(record)

    def list_runs_for_comparison(self, experiment_id: UUID) -> tuple[ComparisonRun, ...]:
        with self._sessions() as session:
            rows = session.execute(
                select(RunModel, RunSummaryModel)
                .outerjoin(RunSummaryModel)
                .where(RunModel.experiment_id == experiment_id)
                .order_by(RunModel.case_id, RunModel.seed)
            )
            result: list[ComparisonRun] = []
            for run, summary in rows:
                summary_record = (
                    None
                    if summary is None
                    else RunSummaryRecord(
                        summary.alive_count,
                        summary.births_total,
                        summary.deaths_total,
                        summary.active_lineage_count,
                        summary.final_total_energy,
                        summary.final_total_resources,
                    )
                )
                result.append(
                    ComparisonRun(
                        run.run_id,
                        run.case_id,
                        run.config_hash,
                        run.seed,
                        run.status,
                        run.mode,
                        summary_record,
                    )
                )
            return tuple(result)

    def _transition(self, run_id: UUID, before: str, after: str, **values: object) -> None:
        with self._sessions.begin() as session:
            run = self._required_run(session, run_id)
            if run.status != before:
                raise InvalidRunTransitionError(f"cannot transition {run.status} to {after}")
            run.status = after
            for key, value in values.items():
                setattr(run, key, value)

    @staticmethod
    def _required_run(session: Session, run_id: UUID) -> RunModel:
        run = session.get(RunModel, run_id)
        if run is None:
            raise RegistryIntegrityError(f"unknown run {run_id}")
        return run

    @staticmethod
    def _experiment_record(model: ExperimentModel) -> ExperimentRecord:
        return ExperimentRecord(
            model.experiment_id,
            model.name,
            model.description,
            model.objective,
            model.hypothesis,
            model.label,
            model.status,
            model.created_at,
            model.updated_at,
        )

    @staticmethod
    def _run_from_manifest(
        manifest: RunManifest, experiment_id: UUID | None, local_path: str | None
    ) -> RunModel:
        return RunModel(
            run_id=UUID(manifest.run_id),
            experiment_id=experiment_id,
            case_id=manifest.case_id,
            batch_ordinal=manifest.batch_ordinal,
            seed=manifest.seed,
            status=manifest.status.value,
            mode=manifest.mode.value,
            config_hash=manifest.config_hash,
            compile_signature=manifest.compile_signature,
            package_version=manifest.code_version.package_version,
            git_commit=manifest.code_version.git_commit,
            git_dirty=manifest.code_version.git_dirty,
            python_version=manifest.environment.python_version,
            jax_version=manifest.environment.jax_version,
            equinox_version=manifest.environment.equinox_version,
            platform=manifest.environment.platform,
            total_steps=manifest.total_steps,
            chunk_steps=manifest.chunk_steps,
            initial_step=manifest.initial_step,
            final_step=manifest.final_step,
            steps_executed=manifest.steps_executed or 0,
            created_at=_timestamp(manifest.created_at_utc),
            started_at=_timestamp(manifest.started_at_utc),
            finished_at=_timestamp(manifest.finished_at_utc),
            local_run_path=local_path,
            error_type=None,
            error_message=None,
            error_step=None,
        )
