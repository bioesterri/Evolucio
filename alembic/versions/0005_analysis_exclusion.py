"""Auditable analytical exclusion independent of run lifecycle."""

import sqlalchemy as sa
from alembic import op

revision = "0005_analysis_exclusion"
down_revision = "0004_run_recovery"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_analysis_exclusion",
        sa.Column("run_id", sa.UUID(), sa.ForeignKey("run.run_id"), primary_key=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("discarded_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("run_analysis_exclusion")
