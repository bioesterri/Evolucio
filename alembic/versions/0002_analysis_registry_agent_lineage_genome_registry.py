"""agent_lineage_genome_registry"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_analysis_registry"
down_revision: str | None = "0001_initial_experiment_registry"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_summary",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("agent_id", sa.BigInteger(), nullable=False),
        sa.Column("parent_agent_id", sa.BigInteger(), nullable=True),
        sa.Column("lineage_id", sa.BigInteger(), nullable=False),
        sa.Column("genome_id", sa.BigInteger(), nullable=False),
        sa.Column("generation", sa.BigInteger(), nullable=False),
        sa.Column("birth_step", sa.BigInteger(), nullable=False),
        sa.Column("death_step", sa.BigInteger(), nullable=True),
        sa.Column("final_age", sa.BigInteger(), nullable=False),
        sa.Column("final_energy", sa.Double(), nullable=False),
        sa.Column("alive_at_end", sa.Boolean(), nullable=False),
        sa.Column("death_cause_code", sa.Integer(), nullable=True),
        sa.Column("death_phase_code", sa.Integer(), nullable=True),
        sa.Column("direct_offspring_count", sa.BigInteger(), nullable=False),
        sa.CheckConstraint(
            "(alive_at_end AND death_step IS NULL AND death_cause_code IS NULL "
            "AND death_phase_code IS NULL) OR (NOT alive_at_end AND death_step IS NOT NULL)",
            name="ck_agent_summary_death",
        ),
        sa.CheckConstraint(
            "(parent_agent_id IS NULL) = (generation = 0)", name="ck_agent_summary_founder"
        ),
        sa.CheckConstraint(
            "generation >= 0 AND birth_step >= 0 AND final_age >= 0 "
            "AND direct_offspring_count >= 0",
            name="ck_agent_summary_nonnegative",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["run.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "agent_id"),
        sa.UniqueConstraint(
            "run_id", "agent_id", "genome_id", name="uq_agent_summary_genome_identity"
        ),
    )
    op.create_index(
        "ix_agent_summary_alive_at_end", "agent_summary", ["run_id", "alive_at_end"], unique=False
    )
    op.create_index(
        "ix_agent_summary_genome_id", "agent_summary", ["run_id", "genome_id"], unique=False
    )
    op.create_index(
        "ix_agent_summary_lineage_id", "agent_summary", ["run_id", "lineage_id"], unique=False
    )
    op.create_index(
        "ix_agent_summary_parent_agent_id",
        "agent_summary",
        ["run_id", "parent_agent_id"],
        unique=False,
    )
    op.create_table(
        "lineage_summary",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("lineage_id", sa.BigInteger(), nullable=False),
        sa.Column("founder_agent_id", sa.BigInteger(), nullable=False),
        sa.Column("founder_genome_id", sa.BigInteger(), nullable=False),
        sa.Column("total_agents", sa.BigInteger(), nullable=False),
        sa.Column("total_descendants", sa.BigInteger(), nullable=False),
        sa.Column("births_total", sa.BigInteger(), nullable=False),
        sa.Column("deaths_total", sa.BigInteger(), nullable=False),
        sa.Column("alive_final", sa.BigInteger(), nullable=False),
        sa.Column("max_generation", sa.BigInteger(), nullable=False),
        sa.Column("founder_birth_step", sa.BigInteger(), nullable=False),
        sa.Column("last_birth_step", sa.BigInteger(), nullable=True),
        sa.Column("extinction_step", sa.BigInteger(), nullable=True),
        sa.Column("duration_steps", sa.BigInteger(), nullable=False),
        sa.Column("final_population_fraction", sa.Double(), nullable=False),
        sa.Column("mutation_selected_total", sa.BigInteger(), nullable=False),
        sa.Column("mutation_effective_total", sa.BigInteger(), nullable=False),
        sa.Column("mutation_abs_delta_total", sa.Double(), nullable=False),
        sa.Column("reproductive_leader_agent_id", sa.BigInteger(), nullable=False),
        sa.Column("reproductive_leader_offspring", sa.BigInteger(), nullable=False),
        sa.Column("survival_leader_agent_id", sa.BigInteger(), nullable=False),
        sa.Column("survival_leader_age", sa.BigInteger(), nullable=False),
        sa.CheckConstraint(
            "max_generation >= 0 AND duration_steps >= 0 "
            "AND final_population_fraction BETWEEN 0 AND 1",
            name="ck_lineage_summary_ranges",
        ),
        sa.CheckConstraint(
            "total_agents >= 1 AND total_descendants = total_agents - 1 "
            "AND births_total = total_descendants AND deaths_total >= 0 AND alive_final >= 0 "
            "AND deaths_total + alive_final = total_agents",
            name="ck_lineage_summary_counts",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["run.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "lineage_id"),
    )
    op.create_table(
        "genome_registry",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("genome_id", sa.BigInteger(), nullable=False),
        sa.Column("genome_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("agent_id", sa.BigInteger(), nullable=False),
        sa.Column("parent_genome_id", sa.BigInteger(), nullable=True),
        sa.Column("lineage_id", sa.BigInteger(), nullable=False),
        sa.Column("generation", sa.BigInteger(), nullable=False),
        sa.Column("parameter_count", sa.Integer(), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("artifact_sha256", sa.CHAR(length=64), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("policy_schema_name", sa.Text(), nullable=False),
        sa.Column("policy_schema_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "genome_hash ~ '^[a-f0-9]{64}$' AND artifact_sha256 ~ '^[a-f0-9]{64}$'",
            name="ck_genome_registry_hashes",
        ),
        sa.CheckConstraint(
            "generation >= 0 AND parameter_count > 0 AND byte_size > 0 "
            "AND policy_schema_version > 0",
            name="ck_genome_registry_ranges",
        ),
        sa.ForeignKeyConstraint(
            ["run_id", "agent_id", "genome_id"],
            ["agent_summary.run_id", "agent_summary.agent_id", "agent_summary.genome_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["run.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "genome_id"),
    )
    op.create_table(
        "genome_highlight",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("genome_id", sa.BigInteger(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("selected_step", sa.BigInteger(), nullable=False),
        sa.CheckConstraint(
            "reason IN ('founder', 'final_lineage_representative') AND selected_step >= 0",
            name="ck_genome_highlight_reason_step",
        ),
        sa.ForeignKeyConstraint(
            ["run_id", "genome_id"],
            ["genome_registry.run_id", "genome_registry.genome_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "genome_id", "reason"),
    )


def downgrade() -> None:
    op.drop_table("genome_highlight")
    op.drop_table("genome_registry")
    op.drop_table("lineage_summary")
    op.drop_index("ix_agent_summary_parent_agent_id", table_name="agent_summary")
    op.drop_index("ix_agent_summary_lineage_id", table_name="agent_summary")
    op.drop_index("ix_agent_summary_genome_id", table_name="agent_summary")
    op.drop_index("ix_agent_summary_alive_at_end", table_name="agent_summary")
    op.drop_table("agent_summary")
