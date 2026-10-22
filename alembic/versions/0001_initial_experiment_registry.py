"""Initial experiment and run metadata registry.

Revision ID: 0001_initial_experiment_registry
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial_experiment_registry"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    timestamp = sa.DateTime(timezone=True)
    uuid = postgresql.UUID(as_uuid=True)
    op.create_table(
        "experiment",
        sa.Column("experiment_id", uuid, primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("objective", sa.Text()),
        sa.Column("hypothesis", sa.Text()),
        sa.Column("label", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", timestamp, nullable=False),
        sa.Column("updated_at", timestamp, nullable=False),
        sa.CheckConstraint("length(btrim(name)) > 0", name="ck_experiment_name_nonempty"),
        sa.CheckConstraint("length(btrim(status)) > 0", name="ck_experiment_status_nonempty"),
    )
    op.create_table(
        "run",
        sa.Column("run_id", uuid, primary_key=True),
        sa.Column(
            "experiment_id", uuid, sa.ForeignKey("experiment.experiment_id", ondelete="RESTRICT")
        ),
        sa.Column("case_id", sa.Text()),
        sa.Column("batch_ordinal", sa.Integer()),
        sa.Column("seed", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("compile_signature", sa.Text()),
        sa.Column("package_version", sa.Text(), nullable=False),
        sa.Column("git_commit", sa.Text()),
        sa.Column("git_dirty", sa.Boolean()),
        sa.Column("python_version", sa.Text()),
        sa.Column("jax_version", sa.Text()),
        sa.Column("equinox_version", sa.Text()),
        sa.Column("platform", sa.Text()),
        sa.Column("total_steps", sa.BigInteger(), nullable=False),
        sa.Column("chunk_steps", sa.Integer(), nullable=False),
        sa.Column("initial_step", sa.BigInteger(), nullable=False),
        sa.Column("final_step", sa.BigInteger()),
        sa.Column("steps_executed", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("created_at", timestamp, nullable=False),
        sa.Column("started_at", timestamp),
        sa.Column("finished_at", timestamp),
        sa.Column("local_run_path", sa.Text()),
        sa.Column("error_type", sa.Text()),
        sa.Column("error_message", sa.Text()),
        sa.Column("error_step", sa.BigInteger()),
        sa.CheckConstraint("total_steps > 0", name="ck_run_total_steps_positive"),
        sa.CheckConstraint("chunk_steps > 0", name="ck_run_chunk_steps_positive"),
        sa.CheckConstraint("steps_executed >= 0", name="ck_run_steps_executed_nonnegative"),
        sa.CheckConstraint(
            "status <> 'completed' OR (finished_at IS NOT NULL AND final_step IS NOT NULL)",
            name="ck_run_completed_fields",
        ),
        sa.CheckConstraint(
            "status <> 'failed' OR (finished_at IS NOT NULL AND error_type IS NOT NULL)",
            name="ck_run_failed_fields",
        ),
    )
    for name, columns in (
        ("ix_run_experiment_id", ["experiment_id"]),
        ("ix_run_config_hash", ["config_hash"]),
        ("ix_run_seed", ["seed"]),
        ("ix_run_status", ["status"]),
        ("ix_run_created_at", ["created_at"]),
        ("ix_run_experiment_config", ["experiment_id", "config_hash"]),
        ("ix_run_config_seed", ["config_hash", "seed"]),
    ):
        op.create_index(name, "run", columns)
    op.create_table(
        "config_snapshot",
        sa.Column(
            "run_id", uuid, sa.ForeignKey("run.run_id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), nullable=False),
        sa.Column("artifact_relative_path", sa.Text(), nullable=False),
        sa.Column("artifact_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", timestamp, nullable=False),
    )
    op.create_table(
        "run_summary",
        sa.Column(
            "run_id", uuid, sa.ForeignKey("run.run_id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("alive_count", sa.BigInteger()),
        sa.Column("births_total", sa.BigInteger()),
        sa.Column("deaths_total", sa.BigInteger()),
        sa.Column("active_lineage_count", sa.BigInteger()),
        sa.Column("final_total_energy", sa.Double()),
        sa.Column("final_total_resources", sa.Double()),
        sa.Column("population_extinct", sa.Boolean()),
        sa.Column("created_at", timestamp, nullable=False),
    )
    op.create_table(
        "artifact",
        sa.Column("artifact_id", uuid, primary_key=True),
        sa.Column("run_id", uuid, sa.ForeignKey("run.run_id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("media_type", sa.Text(), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("schema_name", sa.Text()),
        sa.Column("schema_version", sa.Integer()),
        sa.Column("created_at", timestamp, nullable=False),
        sa.UniqueConstraint("run_id", "relative_path", name="uq_artifact_run_path"),
        sa.CheckConstraint("byte_size >= 0", name="ck_artifact_byte_size_nonnegative"),
    )
    op.create_index("ix_artifact_run_id", "artifact", ["run_id"])
    op.create_index("ix_artifact_kind", "artifact", ["kind"])
    op.create_table(
        "dataset_index",
        sa.Column("dataset_id", uuid, primary_key=True),
        sa.Column("run_id", uuid, sa.ForeignKey("run.run_id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_kind", sa.Text(), nullable=False),
        sa.Column("storage_format", sa.Text(), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("schema_name", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("row_count", sa.BigInteger()),
        sa.Column("part_count", sa.Integer()),
        sa.Column("min_step", sa.BigInteger()),
        sa.Column("max_step", sa.BigInteger()),
        sa.Column("complete", sa.Boolean(), nullable=False),
        sa.Column("index_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", timestamp, nullable=False),
        sa.UniqueConstraint("run_id", "dataset_kind", name="uq_dataset_run_kind"),
    )
    op.create_index("ix_dataset_run_id", "dataset_index", ["run_id"])
    op.create_index("ix_dataset_kind", "dataset_index", ["dataset_kind"])
    op.create_table(
        "state_capture",
        sa.Column("capture_id", uuid, primary_key=True),
        sa.Column("run_id", uuid, sa.ForeignKey("run.run_id", ondelete="CASCADE"), nullable=False),
        sa.Column("capture_kind", sa.Text(), nullable=False),
        sa.Column("state_step", sa.BigInteger(), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("schema_name", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("created_at", timestamp, nullable=False),
        sa.UniqueConstraint(
            "run_id", "capture_kind", "state_step", name="uq_capture_run_kind_step"
        ),
    )
    op.create_index(
        "ix_capture_run_kind_step", "state_capture", ["run_id", "capture_kind", "state_step"]
    )


def downgrade() -> None:
    for table in (
        "state_capture",
        "dataset_index",
        "artifact",
        "run_summary",
        "config_snapshot",
        "run",
        "experiment",
    ):
        op.drop_table(table)
