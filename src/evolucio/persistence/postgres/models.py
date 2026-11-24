"""SQLAlchemy 2.x models for the PostgreSQL metadata registry."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative metadata used by Alembic, never for production create-all."""


class ExperimentModel(Base):
    __tablename__ = "experiment"
    __table_args__ = (
        CheckConstraint("length(btrim(name)) > 0", name="ck_experiment_name_nonempty"),
        CheckConstraint("length(btrim(status)) > 0", name="ck_experiment_status_nonempty"),
    )

    experiment_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    objective: Mapped[str | None] = mapped_column(Text)
    hypothesis: Mapped[str | None] = mapped_column(Text)
    label: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RunModel(Base):
    __tablename__ = "run"
    __table_args__ = (
        CheckConstraint("total_steps > 0", name="ck_run_total_steps_positive"),
        CheckConstraint("chunk_steps > 0", name="ck_run_chunk_steps_positive"),
        CheckConstraint("steps_executed >= 0", name="ck_run_steps_executed_nonnegative"),
        CheckConstraint(
            "status <> 'completed' OR (finished_at IS NOT NULL AND final_step IS NOT NULL)",
            name="ck_run_completed_fields",
        ),
        CheckConstraint(
            "status <> 'failed' OR (finished_at IS NOT NULL AND error_type IS NOT NULL)",
            name="ck_run_failed_fields",
        ),
        Index("ix_run_experiment_config", "experiment_id", "config_hash"),
        Index("ix_run_config_seed", "config_hash", "seed"),
    )

    run_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    experiment_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("experiment.experiment_id", ondelete="RESTRICT"),
        index=True,
    )
    case_id: Mapped[str | None] = mapped_column(Text)
    batch_ordinal: Mapped[int | None] = mapped_column(Integer)
    seed: Mapped[int] = mapped_column(BigInteger, index=True)
    status: Mapped[str] = mapped_column(Text, index=True)
    mode: Mapped[str] = mapped_column(Text)
    config_hash: Mapped[str] = mapped_column(String(64), index=True)
    compile_signature: Mapped[str | None] = mapped_column(Text)
    package_version: Mapped[str] = mapped_column(Text)
    git_commit: Mapped[str | None] = mapped_column(Text)
    git_dirty: Mapped[bool | None] = mapped_column(Boolean)
    python_version: Mapped[str | None] = mapped_column(Text)
    jax_version: Mapped[str | None] = mapped_column(Text)
    equinox_version: Mapped[str | None] = mapped_column(Text)
    platform: Mapped[str | None] = mapped_column(Text)
    total_steps: Mapped[int] = mapped_column(BigInteger)
    chunk_steps: Mapped[int] = mapped_column(Integer)
    initial_step: Mapped[int] = mapped_column(BigInteger)
    final_step: Mapped[int | None] = mapped_column(BigInteger)
    steps_executed: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    local_run_path: Mapped[str | None] = mapped_column(Text)
    error_type: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    error_step: Mapped[int | None] = mapped_column(BigInteger)


class ConfigSnapshotModel(Base):
    __tablename__ = "config_snapshot"

    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("run.run_id", ondelete="CASCADE"), primary_key=True
    )
    config_hash: Mapped[str] = mapped_column(String(64))
    schema_version: Mapped[int] = mapped_column(Integer)
    config_json: Mapped[dict[str, object]] = mapped_column(JSONB)
    artifact_relative_path: Mapped[str] = mapped_column(Text)
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RunSummaryModel(Base):
    __tablename__ = "run_summary"

    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("run.run_id", ondelete="CASCADE"), primary_key=True
    )
    alive_count: Mapped[int | None] = mapped_column(BigInteger)
    births_total: Mapped[int | None] = mapped_column(BigInteger)
    deaths_total: Mapped[int | None] = mapped_column(BigInteger)
    active_lineage_count: Mapped[int | None] = mapped_column(BigInteger)
    final_total_energy: Mapped[float | None] = mapped_column(Double)
    final_total_resources: Mapped[float | None] = mapped_column(Double)
    population_extinct: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ArtifactModel(Base):
    __tablename__ = "artifact"
    __table_args__ = (
        UniqueConstraint("run_id", "relative_path", name="uq_artifact_run_path"),
        CheckConstraint("byte_size >= 0", name="ck_artifact_byte_size_nonnegative"),
        Index("ix_artifact_kind", "kind"),
    )

    artifact_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("run.run_id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text)
    relative_path: Mapped[str] = mapped_column(Text)
    media_type: Mapped[str] = mapped_column(Text)
    byte_size: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    schema_name: Mapped[str | None] = mapped_column(Text)
    schema_version: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DatasetIndexModel(Base):
    __tablename__ = "dataset_index"
    __table_args__ = (
        UniqueConstraint("run_id", "dataset_kind", name="uq_dataset_run_kind"),
        Index("ix_dataset_kind", "dataset_kind"),
        Index("ix_dataset_run_id", "run_id"),
    )

    dataset_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("run.run_id", ondelete="CASCADE")
    )
    dataset_kind: Mapped[str] = mapped_column(Text)
    storage_format: Mapped[str] = mapped_column(Text)
    relative_path: Mapped[str] = mapped_column(Text)
    schema_name: Mapped[str] = mapped_column(Text)
    schema_version: Mapped[int] = mapped_column(Integer)
    row_count: Mapped[int | None] = mapped_column(BigInteger)
    part_count: Mapped[int | None] = mapped_column(Integer)
    min_step: Mapped[int | None] = mapped_column(BigInteger)
    max_step: Mapped[int | None] = mapped_column(BigInteger)
    complete: Mapped[bool] = mapped_column(Boolean)
    index_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class StateCaptureModel(Base):
    __tablename__ = "state_capture"
    __table_args__ = (
        UniqueConstraint("run_id", "capture_kind", "state_step", name="uq_capture_run_kind_step"),
        Index("ix_capture_run_kind_step", "run_id", "capture_kind", "state_step"),
    )

    capture_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("run.run_id", ondelete="CASCADE")
    )
    capture_kind: Mapped[str] = mapped_column(Text)
    state_step: Mapped[int] = mapped_column(BigInteger)
    relative_path: Mapped[str] = mapped_column(Text)
    schema_name: Mapped[str] = mapped_column(Text)
    schema_version: Mapped[int] = mapped_column(Integer)
    byte_size: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AgentSummaryModel(Base):
    __tablename__ = "agent_summary"
    __table_args__ = (
        CheckConstraint(
            "generation >= 0 AND birth_step >= 0 AND final_age >= 0 "
            "AND direct_offspring_count >= 0",
            name="ck_agent_summary_nonnegative",
        ),
        CheckConstraint(
            "(alive_at_end AND death_step IS NULL AND death_cause_code IS NULL "
            "AND death_phase_code IS NULL) OR (NOT alive_at_end AND death_step IS NOT NULL)",
            name="ck_agent_summary_death",
        ),
        CheckConstraint(
            "(parent_agent_id IS NULL) = (generation = 0)", name="ck_agent_summary_founder"
        ),
        UniqueConstraint(
            "run_id", "agent_id", "genome_id", name="uq_agent_summary_genome_identity"
        ),
        Index("ix_agent_summary_lineage_id", "run_id", "lineage_id"),
        Index("ix_agent_summary_parent_agent_id", "run_id", "parent_agent_id"),
        Index("ix_agent_summary_genome_id", "run_id", "genome_id"),
        Index("ix_agent_summary_alive_at_end", "run_id", "alive_at_end"),
    )

    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("run.run_id", ondelete="CASCADE"), primary_key=True
    )
    agent_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    parent_agent_id: Mapped[int | None] = mapped_column(BigInteger)
    lineage_id: Mapped[int] = mapped_column(BigInteger)
    genome_id: Mapped[int] = mapped_column(BigInteger)
    generation: Mapped[int] = mapped_column(BigInteger)
    birth_step: Mapped[int] = mapped_column(BigInteger)
    death_step: Mapped[int | None] = mapped_column(BigInteger)
    final_age: Mapped[int] = mapped_column(BigInteger)
    final_energy: Mapped[float] = mapped_column(Double)
    alive_at_end: Mapped[bool] = mapped_column(Boolean)
    death_cause_code: Mapped[int | None] = mapped_column(Integer)
    death_phase_code: Mapped[int | None] = mapped_column(Integer)
    direct_offspring_count: Mapped[int] = mapped_column(BigInteger)


class LineageSummaryModel(Base):
    __tablename__ = "lineage_summary"
    __table_args__ = (
        CheckConstraint(
            "total_agents >= 1 AND total_descendants = total_agents - 1 "
            "AND births_total = total_descendants AND deaths_total >= 0 AND alive_final >= 0 "
            "AND deaths_total + alive_final = total_agents",
            name="ck_lineage_summary_counts",
        ),
        CheckConstraint(
            "max_generation >= 0 AND duration_steps >= 0 "
            "AND final_population_fraction BETWEEN 0 AND 1",
            name="ck_lineage_summary_ranges",
        ),
    )

    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("run.run_id", ondelete="CASCADE"), primary_key=True
    )
    lineage_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    founder_agent_id: Mapped[int] = mapped_column(BigInteger)
    founder_genome_id: Mapped[int] = mapped_column(BigInteger)
    total_agents: Mapped[int] = mapped_column(BigInteger)
    total_descendants: Mapped[int] = mapped_column(BigInteger)
    births_total: Mapped[int] = mapped_column(BigInteger)
    deaths_total: Mapped[int] = mapped_column(BigInteger)
    alive_final: Mapped[int] = mapped_column(BigInteger)
    max_generation: Mapped[int] = mapped_column(BigInteger)
    founder_birth_step: Mapped[int] = mapped_column(BigInteger)
    last_birth_step: Mapped[int | None] = mapped_column(BigInteger)
    extinction_step: Mapped[int | None] = mapped_column(BigInteger)
    duration_steps: Mapped[int] = mapped_column(BigInteger)
    final_population_fraction: Mapped[float] = mapped_column(Double)
    mutation_selected_total: Mapped[int] = mapped_column(BigInteger)
    mutation_effective_total: Mapped[int] = mapped_column(BigInteger)
    mutation_abs_delta_total: Mapped[float] = mapped_column(Double)
    reproductive_leader_agent_id: Mapped[int] = mapped_column(BigInteger)
    reproductive_leader_offspring: Mapped[int] = mapped_column(BigInteger)
    survival_leader_agent_id: Mapped[int] = mapped_column(BigInteger)
    survival_leader_age: Mapped[int] = mapped_column(BigInteger)


class GenomeRegistryModel(Base):
    __tablename__ = "genome_registry"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "agent_id", "genome_id"],
            ["agent_summary.run_id", "agent_summary.agent_id", "agent_summary.genome_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "genome_hash ~ '^[a-f0-9]{64}$' AND artifact_sha256 ~ '^[a-f0-9]{64}$'",
            name="ck_genome_registry_hashes",
        ),
        CheckConstraint(
            "generation >= 0 AND parameter_count > 0 AND byte_size > 0 "
            "AND policy_schema_version > 0",
            name="ck_genome_registry_ranges",
        ),
    )

    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("run.run_id", ondelete="CASCADE"), primary_key=True
    )
    genome_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    genome_hash: Mapped[str] = mapped_column(CHAR(64))
    agent_id: Mapped[int] = mapped_column(BigInteger)
    parent_genome_id: Mapped[int | None] = mapped_column(BigInteger)
    lineage_id: Mapped[int] = mapped_column(BigInteger)
    generation: Mapped[int] = mapped_column(BigInteger)
    parameter_count: Mapped[int] = mapped_column(Integer)
    relative_path: Mapped[str] = mapped_column(Text)
    artifact_sha256: Mapped[str] = mapped_column(CHAR(64))
    byte_size: Mapped[int] = mapped_column(BigInteger)
    policy_schema_name: Mapped[str] = mapped_column(Text)
    policy_schema_version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class GenomeHighlightModel(Base):
    __tablename__ = "genome_highlight"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "genome_id"],
            ["genome_registry.run_id", "genome_registry.genome_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "reason IN ('founder', 'final_lineage_representative') AND selected_step >= 0",
            name="ck_genome_highlight_reason_step",
        ),
    )

    run_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    genome_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    reason: Mapped[str] = mapped_column(Text, primary_key=True)
    selected_step: Mapped[int] = mapped_column(BigInteger)


class MlflowExperimentBindingModel(Base):
    __tablename__ = "mlflow_experiment_binding"

    experiment_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("experiment.experiment_id", ondelete="CASCADE"),
        primary_key=True,
    )
    mlflow_experiment_id: Mapped[str] = mapped_column(Text, unique=True)
    mlflow_experiment_name: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MlflowRunBindingModel(Base):
    __tablename__ = "mlflow_run_binding"
    __table_args__ = (
        CheckConstraint(
            "sync_status IN ('active', 'synced', 'degraded', 'failed')",
            name="ck_mlflow_run_binding_sync_status",
        ),
    )

    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("run.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    mlflow_run_id: Mapped[str] = mapped_column(Text, unique=True)
    mlflow_experiment_id: Mapped[str] = mapped_column(Text)
    sync_status: Mapped[str] = mapped_column(Text)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RunRecoveryModel(Base):
    __tablename__ = "run_recovery"
    __table_args__ = (
        CheckConstraint(
            "run_id <> source_run_id AND run_id <> root_run_id", name="ck_run_recovery_identity"
        ),
        CheckConstraint(
            "recovery_depth > 0 AND target_step > source_checkpoint_step",
            name="ck_run_recovery_horizon",
        ),
    )

    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("run.run_id"), primary_key=True
    )
    source_run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("run.run_id"), index=True
    )
    root_run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("run.run_id"), index=True
    )
    source_checkpoint_step: Mapped[int] = mapped_column(BigInteger)
    source_checkpoint_hash: Mapped[str] = mapped_column(String(64))
    inherited_steps: Mapped[int] = mapped_column(BigInteger)
    target_step: Mapped[int] = mapped_column(BigInteger)
    recovery_depth: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
