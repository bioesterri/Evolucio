"""Run recovery provenance."""

import sqlalchemy as sa
from alembic import op

revision = "0004_run_recovery"
down_revision = "0003_mlflow_tracking_bindings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_recovery",
        sa.Column("run_id", sa.UUID(), sa.ForeignKey("run.run_id"), primary_key=True),
        sa.Column("source_run_id", sa.UUID(), sa.ForeignKey("run.run_id"), nullable=False),
        sa.Column("root_run_id", sa.UUID(), sa.ForeignKey("run.run_id"), nullable=False),
        sa.Column("source_checkpoint_step", sa.BigInteger(), nullable=False),
        sa.Column("source_checkpoint_hash", sa.String(64), nullable=False),
        sa.Column("inherited_steps", sa.BigInteger(), nullable=False),
        sa.Column("target_step", sa.BigInteger(), nullable=False),
        sa.Column("recovery_depth", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "run_id <> source_run_id AND run_id <> root_run_id", name="ck_run_recovery_identity"
        ),
        sa.CheckConstraint(
            "recovery_depth > 0 AND target_step > source_checkpoint_step",
            name="ck_run_recovery_horizon",
        ),
    )
    op.create_index("ix_run_recovery_source_run_id", "run_recovery", ["source_run_id"])
    op.create_index("ix_run_recovery_root_run_id", "run_recovery", ["root_run_id"])


def downgrade() -> None:
    op.drop_table("run_recovery")
