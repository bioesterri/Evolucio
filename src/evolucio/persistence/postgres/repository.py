"""Transactional repository for experiments, runs and artifact indexes."""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID, uuid4

from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from evolucio.config.freeze import canonical_json_and_hash
from evolucio.persistence.local.artifacts import (
    ArtifactRecord,
    validate_relative_path,
    validate_sha256,
)
from evolucio.persistence.local.manifest import RunManifest, RunStatus
from evolucio.persistence.parquet.index import INDEX_PATH as PARQUET_INDEX_PATH
from evolucio.recovery.provenance import RecoveryProvenance

from .errors import (
    ArtifactIndexConflictError,
    DatabaseUnavailableError,
    InvalidRunTransitionError,
    RegistryIntegrityError,
    RunAlreadyExistsError,
)
from .models import (
    AgentSummaryModel,
    ArtifactModel,
    ConfigSnapshotModel,
    DatasetIndexModel,
    ExperimentModel,
    GenomeHighlightModel,
    GenomeRegistryModel,
    LineageSummaryModel,
    MlflowExperimentBindingModel,
    MlflowRunBindingModel,
    RunModel,
    RunRecoveryModel,
    RunSummaryModel,
    StateCaptureModel,
)
from .types import ComparisonRun, ExperimentRecord, GenomeHighlights, RunIndexes, RunSummaryRecord


def _now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("registry timestamps must include a timezone")
    return value.astimezone(UTC)


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

    @contextmanager
    def _session(self, *, write: bool = False) -> Generator[Session]:
        try:
            with self._sessions.begin() if write else self._sessions() as session:
                yield session
        except (OperationalError, InterfaceError) as exc:
            raise DatabaseUnavailableError("PostgreSQL registry connection failed") from exc
        except IntegrityError as exc:
            raise RegistryIntegrityError("PostgreSQL relational constraint failed") from exc
        except SQLAlchemyError as exc:
            raise RegistryIntegrityError("PostgreSQL registry operation failed") from exc

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
        with self._session(write=True) as session:
            session.add(model)
        return self._experiment_record(model)

    def get_experiment(self, experiment_id: UUID) -> ExperimentRecord | None:
        with self._session() as session:
            model = session.get(ExperimentModel, experiment_id)
            return None if model is None else self._experiment_record(model)

    def list_experiments(self) -> tuple[ExperimentRecord, ...]:
        with self._session() as session:
            models = session.scalars(select(ExperimentModel).order_by(ExperimentModel.created_at))
            return tuple(self._experiment_record(model) for model in models)

    def get_mlflow_experiment_binding(
        self, experiment_id: UUID
    ) -> MlflowExperimentBindingModel | None:
        with self._session() as session:
            return session.get(MlflowExperimentBindingModel, experiment_id)

    def get_mlflow_run_binding(self, run_id: UUID) -> MlflowRunBindingModel | None:
        with self._session() as session:
            return session.get(MlflowRunBindingModel, run_id)

    @contextmanager
    def lock_mlflow_experiment(
        self, experiment_id: UUID
    ) -> Generator[tuple[Session, ExperimentModel]]:
        with self._session(write=True) as session:
            experiment = session.get(ExperimentModel, experiment_id, with_for_update=True)
            if experiment is None:
                raise RegistryIntegrityError("experiment does not exist")
            yield session, experiment

    @contextmanager
    def lock_mlflow_run(self, run_id: UUID) -> Generator[Session]:
        with self._session(write=True) as session:
            if session.get(RunModel, run_id, with_for_update=True) is None:
                raise RegistryIntegrityError("run does not exist")
            yield session

    def create_run(
        self,
        *,
        manifest: RunManifest,
        config_json: dict[str, object],
        config_schema_version: int,
        config_artifact_sha256: str,
        experiment_id: UUID | None = None,
        local_run_path: str | None = None,
        recovery: RecoveryProvenance | None = None,
    ) -> RunModel:
        """Atomically create the PR-35 run identity and immutable config snapshot."""
        if manifest.status is not RunStatus.CREATED:
            raise ValueError("only a created manifest can be registered")
        validate_sha256(config_artifact_sha256)
        config_artifact = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.relative_path == manifest.config_artifact
        )
        if config_artifact.sha256 != config_artifact_sha256:
            raise RegistryIntegrityError("config artifact digest differs from the manifest")
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
            with self._session(write=True) as session:
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
                if recovery is not None:
                    source_id = UUID(recovery.source_run_id)
                    source = self._required_run(session, source_id)
                    parent = session.get(RunRecoveryModel, source_id)
                    root = source_id if parent is None else parent.root_run_id
                    depth = 1 if parent is None else parent.recovery_depth + 1
                    validate_sha256(recovery.source_checkpoint_hash)
                    if (
                        source_id == run.run_id
                        or source.status not in {"failed", "stopped"}
                        or (source.experiment_id, source.case_id, source.seed, source.config_hash)
                        != (run.experiment_id, run.case_id, run.seed, run.config_hash)
                        or source.config_hash != recovery.source_config_hash
                        or root != UUID(recovery.root_run_id)
                        or depth != recovery.recovery_depth
                        or recovery.target_step != source.initial_step + source.total_steps
                        or recovery.source_checkpoint_step != run.initial_step
                        or run.initial_step + run.total_steps != recovery.target_step
                    ):
                        raise RegistryIntegrityError("inconsistent recovery lineage")
                    session.add(
                        RunRecoveryModel(
                            run_id=run.run_id,
                            source_run_id=source_id,
                            root_run_id=root,
                            source_checkpoint_step=recovery.source_checkpoint_step,
                            source_checkpoint_hash=recovery.source_checkpoint_hash,
                            inherited_steps=recovery.inherited_steps,
                            target_step=recovery.target_step,
                            recovery_depth=depth,
                            created_at=run.created_at,
                        )
                    )
        except RegistryIntegrityError as exc:
            if isinstance(exc.__cause__, IntegrityError):
                with self._session() as session:
                    if session.get(RunModel, run.run_id) is not None:
                        raise RunAlreadyExistsError(
                            f"run {manifest.run_id} already exists"
                        ) from exc
            raise
        return run

    def get_run(self, run_id: UUID) -> RunModel | None:
        with self._session() as session:
            return session.get(RunModel, run_id)

    def get_config_snapshot(self, run_id: UUID) -> ConfigSnapshotModel | None:
        with self._session() as session:
            return session.get(ConfigSnapshotModel, run_id)

    def get_run_summary(self, run_id: UUID) -> RunSummaryModel | None:
        with self._session() as session:
            return session.get(RunSummaryModel, run_id)

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
        with self._session() as session:
            return tuple(session.scalars(query.order_by(RunModel.created_at)))

    def mark_run_running(self, run_id: UUID, *, started_at: datetime | None = None) -> None:
        self._transition(
            run_id,
            "created",
            "running",
            started_at=_as_utc(started_at) if started_at is not None else _now(),
        )

    def mark_run_stopped(self, manifest: RunManifest, *, indexes: RunIndexes | None = None) -> None:
        if manifest.status is not RunStatus.STOPPED:
            raise ValueError("a stopped manifest is required")
        with self._session(write=True) as session:
            run = self._required_run(session, UUID(manifest.run_id))
            if run.status not in {"created", "running", "stopped"}:
                raise InvalidRunTransitionError(f"cannot stop run in status {run.status}")
            if (run.config_hash, run.seed) != (manifest.config_hash, manifest.seed):
                raise RegistryIntegrityError("manifest and registry identity differ")
            if run.status == "stopped" and (
                run.finished_at != _timestamp(manifest.finished_at_utc)
                or run.final_step != manifest.final_step
                or run.steps_executed != (manifest.steps_executed or 0)
            ):
                raise RegistryIntegrityError("conflicting terminal stop details")
            if indexes is not None:
                self._sync_indexes(session, run, indexes)
            run.status = "stopped"
            run.finished_at = _timestamp(manifest.finished_at_utc)
            run.final_step = manifest.final_step
            run.steps_executed = manifest.steps_executed or 0

    def list_failed_runs(self) -> tuple[RunModel, ...]:
        return self.list_runs(status="failed")

    def list_stopped_runs(self) -> tuple[RunModel, ...]:
        return self.list_runs(status="stopped")

    def list_recovery_attempts(self, root_run_id: UUID) -> tuple[RunRecoveryModel, ...]:
        with self._session() as session:
            return tuple(
                session.scalars(
                    select(RunRecoveryModel)
                    .where(RunRecoveryModel.root_run_id == root_run_id)
                    .order_by(RunRecoveryModel.created_at, RunRecoveryModel.run_id)
                )
            )

    def get_recovery(self, run_id: UUID) -> RunRecoveryModel | None:
        with self._session() as session:
            return session.get(RunRecoveryModel, run_id)

    def get_recovery_chain(self, run_id: UUID) -> tuple[RunRecoveryModel, ...]:
        chain: list[RunRecoveryModel] = []
        seen: set[UUID] = set()
        with self._session() as session:
            while edge := session.get(RunRecoveryModel, run_id):
                if run_id in seen:
                    raise RegistryIntegrityError("cyclic recovery lineage")
                seen.add(run_id)
                chain.append(edge)
                run_id = edge.source_run_id
        return tuple(reversed(chain))

    def mark_run_failed(
        self,
        run_id: UUID,
        *,
        error_type: str,
        error_message: str,
        error_step: int | None = None,
        finished_at: datetime | None = None,
        indexes: RunIndexes | None = None,
    ) -> None:
        if not error_type:
            raise ValueError("error_type must be nonempty")
        with self._session(write=True) as session:
            run = self._required_run(session, run_id)
            if run.status not in {"created", "running"}:
                raise InvalidRunTransitionError(f"cannot fail run in status {run.status}")
            if indexes is not None:
                self._sync_indexes(session, run, indexes)
            run.status = "failed"
            run.finished_at = _as_utc(finished_at) if finished_at is not None else _now()
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
        indexes: RunIndexes | None = None,
        analysis_run_dir: Path | None = None,
    ) -> None:
        """Atomically install one immutable summary and close the run lifecycle."""
        with self._session(write=True) as session:
            run = self._required_run(session, run_id)
            if run.status != "running":
                raise InvalidRunTransitionError(f"cannot complete run in status {run.status}")
            if session.get(RunSummaryModel, run_id) is not None:
                raise RegistryIntegrityError("run summary is immutable")
            if indexes is not None:
                self._sync_indexes(session, run, indexes)
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
            if not {
                "config.json",
                "summary.json",
                PARQUET_INDEX_PATH,
            } <= artifact_paths or dataset_kinds != {
                "metrics",
                "births",
                "deaths",
                "genealogy",
            }:
                raise RegistryIntegrityError(
                    "completed run requires verified summary/config and complete dataset indexes"
                )
            if analysis_run_dir is not None:
                from .analysis import insert_run_analysis

                insert_run_analysis(session, run_id, analysis_run_dir)
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
            run.status = "completed"
            run.finished_at = _as_utc(finished_at) if finished_at is not None else _now()
            run.final_step, run.steps_executed = final_step, steps_executed

    def index_artifact(self, run_id: UUID, artifact: ArtifactRecord) -> None:
        with self._session(write=True) as session:
            self._required_run(session, run_id)
            self._index_artifact(session, run_id, artifact)

    @staticmethod
    def _index_artifact(session: Session, run_id: UUID, artifact: ArtifactRecord) -> None:
        existing = session.scalar(
            select(ArtifactModel).where(
                ArtifactModel.run_id == run_id,
                ArtifactModel.relative_path == artifact.relative_path,
            )
        )
        if existing is not None:
            if any(
                getattr(existing, field) != getattr(artifact, field)
                for field in ("sha256", "byte_size", "name", "kind", "media_type")
            ):
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
        with self._session() as session:
            return tuple(
                session.scalars(select(ArtifactModel).where(ArtifactModel.run_id == run_id))
            )

    def list_datasets(self, run_id: UUID) -> tuple[DatasetIndexModel, ...]:
        with self._session() as session:
            return tuple(
                session.scalars(select(DatasetIndexModel).where(DatasetIndexModel.run_id == run_id))
            )

    def list_state_captures(self, run_id: UUID) -> tuple[StateCaptureModel, ...]:
        with self._session() as session:
            return tuple(
                session.scalars(select(StateCaptureModel).where(StateCaptureModel.run_id == run_id))
            )

    def index_dataset(self, record: DatasetIndexModel) -> None:
        with self._session(write=True) as session:
            self._required_run(session, record.run_id)
            self._index_dataset(session, record)

    @staticmethod
    def _index_dataset(session: Session, record: DatasetIndexModel) -> None:
        validate_sha256(record.index_sha256)
        validate_relative_path(record.relative_path)
        record.created_at = _as_utc(record.created_at)
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
                    "storage_format",
                    "schema_name",
                    "schema_version",
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
        with self._session(write=True) as session:
            self._required_run(session, record.run_id)
            self._index_state_capture(session, record)

    @staticmethod
    def _index_state_capture(session: Session, record: StateCaptureModel) -> None:
        validate_sha256(record.sha256)
        validate_relative_path(record.relative_path)
        record.created_at = _as_utc(record.created_at)
        existing = session.scalar(
            select(StateCaptureModel).where(
                StateCaptureModel.run_id == record.run_id,
                StateCaptureModel.capture_kind == record.capture_kind,
                StateCaptureModel.state_step == record.state_step,
            )
        )
        if existing is not None:
            if any(
                getattr(existing, field) != getattr(record, field)
                for field in (
                    "sha256",
                    "relative_path",
                    "schema_name",
                    "schema_version",
                    "byte_size",
                )
            ):
                raise ArtifactIndexConflictError(record.relative_path)
            return
        session.add(record)

    def sync_indexes(self, run_id: UUID, indexes: RunIndexes) -> None:
        with self._session(write=True) as session:
            self._sync_indexes(session, self._required_run(session, run_id), indexes)

    def _sync_indexes(self, session: Session, run: RunModel, indexes: RunIndexes) -> None:
        snapshot = session.get(ConfigSnapshotModel, run.run_id)
        if (
            snapshot is None
            or snapshot.config_hash != run.config_hash
            or run.config_hash != indexes.config_hash
            or canonical_json_and_hash(snapshot.config_json)[1] != run.config_hash
        ):
            raise RegistryIntegrityError("local run identity/config differs from PostgreSQL")
        for artifact in indexes.artifacts:
            if (
                artifact.relative_path == snapshot.artifact_relative_path
                and artifact.sha256 != snapshot.artifact_sha256
            ):
                raise ArtifactIndexConflictError(artifact.relative_path)
            self._index_artifact(session, run.run_id, artifact)
        for dataset in indexes.datasets:
            if dataset.run_id != run.run_id:
                raise RegistryIntegrityError("dataset belongs to a different run")
            self._index_dataset(session, dataset)
        for capture in indexes.captures:
            if capture.run_id != run.run_id:
                raise RegistryIntegrityError("state capture belongs to a different run")
            self._index_state_capture(session, capture)

    def list_runs_for_comparison(
        self, experiment_id: UUID, *, selected_runs: frozenset[UUID] = frozenset()
    ) -> tuple[ComparisonRun, ...]:
        with self._session() as session:
            rows = session.execute(
                select(RunModel, RunSummaryModel, RunRecoveryModel)
                .outerjoin(RunSummaryModel, RunSummaryModel.run_id == RunModel.run_id)
                .outerjoin(RunRecoveryModel, RunRecoveryModel.run_id == RunModel.run_id)
                .where(RunModel.experiment_id == experiment_id)
                .order_by(RunModel.case_id, RunModel.seed)
            )
            result: list[ComparisonRun] = []
            for run, summary, recovery in rows:
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
                        run.run_id if recovery is None else recovery.root_run_id,
                        0 if recovery is None else recovery.recovery_depth,
                    )
                )
            completed_by_root: dict[UUID, list[UUID]] = {}
            for entry in result:
                if entry.status == "completed":
                    completed_by_root.setdefault(entry.root_run_id or entry.run_id, []).append(
                        entry.run_id
                    )
            exclusions: set[UUID] = set()
            for root, ids in completed_by_root.items():
                if len(ids) > 1:
                    choices = selected_runs.intersection(ids)
                    if len(choices) != 1:
                        raise RegistryIntegrityError(
                            f"select one completed continuation for trajectory {root}: {ids}"
                        )
                    exclusions.update(set(ids) - choices)
            return tuple(entry for entry in result if entry.run_id not in exclusions)

    def get_agent_summary(self, run_id: UUID, agent_id: int) -> AgentSummaryModel | None:
        with self._sessions() as session:
            return session.get(AgentSummaryModel, (run_id, agent_id))

    def list_agents_by_lineage(
        self, run_id: UUID, lineage_id: int
    ) -> tuple[AgentSummaryModel, ...]:
        with self._sessions() as session:
            return tuple(
                session.scalars(
                    select(AgentSummaryModel)
                    .where(
                        AgentSummaryModel.run_id == run_id,
                        AgentSummaryModel.lineage_id == lineage_id,
                    )
                    .order_by(AgentSummaryModel.agent_id)
                )
            )

    def get_lineage_summary(self, run_id: UUID, lineage_id: int) -> LineageSummaryModel | None:
        with self._sessions() as session:
            return session.get(LineageSummaryModel, (run_id, lineage_id))

    def list_lineages(
        self,
        run_id: UUID,
        *,
        order_by: Literal[
            "lineage_id", "total_descendants", "alive_final", "max_generation"
        ] = "lineage_id",
    ) -> tuple[LineageSummaryModel, ...]:
        order = {
            "lineage_id": LineageSummaryModel.lineage_id,
            "total_descendants": LineageSummaryModel.total_descendants.desc(),
            "alive_final": LineageSummaryModel.alive_final.desc(),
            "max_generation": LineageSummaryModel.max_generation.desc(),
        }[order_by]
        with self._sessions() as session:
            return tuple(
                session.scalars(
                    select(LineageSummaryModel)
                    .where(LineageSummaryModel.run_id == run_id)
                    .order_by(order, LineageSummaryModel.lineage_id)
                )
            )

    def get_highlighted_genomes(
        self,
        run_id: UUID,
    ) -> tuple[GenomeHighlights, ...]:
        return self._highlight_rows(run_id, None)

    def get_genomes_by_reason(
        self,
        run_id: UUID,
        reason: str,
    ) -> tuple[GenomeHighlights, ...]:
        return self._highlight_rows(run_id, reason)

    def _highlight_rows(
        self,
        run_id: UUID,
        reason: str | None,
    ) -> tuple[GenomeHighlights, ...]:
        query = (
            select(GenomeRegistryModel, GenomeHighlightModel)
            .join(
                GenomeHighlightModel,
            )
            .where(GenomeRegistryModel.run_id == run_id)
        )
        if reason is not None:
            query = query.where(GenomeHighlightModel.reason == reason)
        with self._sessions() as session:
            rows = session.execute(
                query.order_by(GenomeRegistryModel.genome_id, GenomeHighlightModel.reason),
            ).tuples()
            grouped: dict[int, tuple[GenomeRegistryModel, list[GenomeHighlightModel]]] = {}
            for genome, highlight in rows:
                grouped.setdefault(genome.genome_id, (genome, []))[1].append(highlight)
            return tuple(
                GenomeHighlights(genome, tuple(highlights))
                for genome, highlights in grouped.values()
            )

    def lineage_outcomes_for_experiment(
        self,
        experiment_id: UUID,
        *,
        selected_runs: frozenset[UUID] = frozenset(),
    ) -> tuple[tuple[UUID, str | None, int, int, int | None, int | None], ...]:
        completed_ids = [
            run.run_id
            for run in self.list_runs_for_comparison(experiment_id, selected_runs=selected_runs)
            if run.status == "completed"
        ]
        with self._sessions() as session:
            rows = session.execute(
                select(
                    RunModel.run_id,
                    RunModel.case_id,
                    RunModel.seed,
                    func.count(LineageSummaryModel.lineage_id),
                    func.max(LineageSummaryModel.max_generation),
                    func.max(LineageSummaryModel.total_agents),
                )
                .outerjoin(LineageSummaryModel)
                .where(
                    RunModel.run_id.in_(completed_ids),
                )
                .group_by(RunModel.run_id)
                .order_by(RunModel.case_id, RunModel.seed, RunModel.run_id)
            )
            return tuple(
                (run_id, case_id, seed, count, generation, size)
                for run_id, case_id, seed, count, generation, size in rows
            )

    def _transition(self, run_id: UUID, before: str, after: str, **values: object) -> None:
        with self._session(write=True) as session:
            run = self._required_run(session, run_id)
            if run.status != before:
                raise InvalidRunTransitionError(f"cannot transition {run.status} to {after}")
            run.status = after
            for key, value in values.items():
                setattr(run, key, value)

    @staticmethod
    def _required_run(session: Session, run_id: UUID) -> RunModel:
        run = session.get(RunModel, run_id, with_for_update=True)
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
