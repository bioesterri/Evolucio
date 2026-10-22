"""SQLAlchemy 2.x models for the PostgreSQL metadata registry."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
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
    )

    dataset_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("run.run_id", ondelete="CASCADE"), index=True
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
