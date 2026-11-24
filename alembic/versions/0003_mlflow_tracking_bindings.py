"""MLflow experiment and run bindings."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_mlflow_tracking_bindings"
down_revision: str | None = "0002_analysis_registry"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "mlflow_experiment_binding",
        sa.Column(
            "experiment_id",
            sa.UUID(),
            sa.ForeignKey("experiment.experiment_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("mlflow_experiment_id", sa.Text(), nullable=False, unique=True),
        sa.Column("mlflow_experiment_name", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "mlflow_run_binding",
        sa.Column(
            "run_id",
            sa.UUID(),
            sa.ForeignKey("run.run_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("mlflow_run_id", sa.Text(), nullable=False, unique=True),
        sa.Column("mlflow_experiment_id", sa.Text(), nullable=False),
        sa.Column("sync_status", sa.Text(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "sync_status IN ('active', 'synced', 'degraded', 'failed')",
            name="ck_mlflow_run_binding_sync_status",
        ),
    )


def downgrade() -> None:
    op.drop_table("mlflow_run_binding")
    op.drop_table("mlflow_experiment_binding")
