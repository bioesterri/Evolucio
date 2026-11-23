"""Versioned lifecycle and lineage projections."""

from dataclasses import dataclass, fields

import polars as pl

ANALYSIS_SUMMARY_SCHEMA_VERSION = 1
AGENT_SUMMARY_SCHEMA_VERSION = 1
LINEAGE_SUMMARY_SCHEMA_VERSION = 1
AGENT_SUMMARY_SCHEMA_NAME = "agent_lifecycle_summary_v1"
LINEAGE_SUMMARY_SCHEMA_NAME = "lineage_outcome_summary_v1"
FOUNDERS_PATH = "artifacts/analysis/founders.parquet"
AGENTS_PATH = "artifacts/analysis/agent_summary.parquet"
LINEAGES_PATH = "artifacts/analysis/lineage_summary.parquet"


@dataclass(frozen=True, slots=True)
class FounderRecord:
    agent_id: int
    lineage_id: int
    genome_id: int
    generation: int
    birth_step: int
    initial_energy: float
    position_x: int
    position_y: int


@dataclass(frozen=True, slots=True)
class AgentSummary:
    run_id: str
    agent_id: int
    parent_agent_id: int | None
    lineage_id: int
    genome_id: int
    generation: int
    birth_step: int
    death_step: int | None
    final_age: int
    final_energy: float
    alive_at_end: bool
    death_cause_code: int | None
    death_phase_code: int | None
    direct_offspring_count: int


@dataclass(frozen=True, slots=True)
class LineageSummary:
    run_id: str
    lineage_id: int
    founder_agent_id: int
    founder_genome_id: int
    total_agents: int
    total_descendants: int
    births_total: int
    deaths_total: int
    alive_final: int
    max_generation: int
    founder_birth_step: int
    last_birth_step: int | None
    extinction_step: int | None
    duration_steps: int
    final_population_fraction: float
    mutation_selected_total: int
    mutation_effective_total: int
    mutation_abs_delta_total: float
    reproductive_leader_agent_id: int
    reproductive_leader_offspring: int
    survival_leader_agent_id: int
    survival_leader_age: int


def _schema(record: type) -> pl.Schema:
    return pl.Schema(
        {
            field.name: (
                pl.String
                if field.type is str
                else pl.Boolean
                if field.type is bool
                else pl.Float64
                if field.type is float
                else pl.Int32
                if field.name.endswith("_code")
                else pl.Int64
            )
            for field in fields(record)
        }
    )


FOUNDER_SCHEMA = _schema(FounderRecord)
AGENT_SCHEMA = _schema(AgentSummary)
LINEAGE_SCHEMA = _schema(LineageSummary)

_TELEMETRY_NULL_PARENT = -1
_IDENTITY = ["agent_id", "parent_agent_id", "lineage_id", "genome_id", "generation"]
SURVIVOR_SCHEMA = pl.Schema(
    {
        name: AGENT_SCHEMA[name]
        for name in [
            *_IDENTITY,
            "birth_step",
            "final_age",
            "final_energy",
        ]
    }
)


def build_agent_summaries(
    run_id: str,
    founders: pl.LazyFrame,
    births: pl.LazyFrame,
    deaths: pl.LazyFrame,
    survivors: pl.LazyFrame,
    final_step: int,
) -> pl.DataFrame:
    errors = "inconsistent agent origins, genealogy or lifecycle endpoints"
    origin = pl.concat(
        [
            founders.with_columns(pl.lit(None, dtype=pl.Int64).alias("parent_agent_id")).select(
                *_IDENTITY, "birth_step"
            ),
            births.select(
                pl.col("child_agent_id").alias("agent_id"),
                "parent_agent_id",
                "lineage_id",
                pl.col("child_genome_id").alias("genome_id"),
                "generation",
                "birth_step",
            ),
        ]
    ).collect()
    parent = origin.lazy().select(
        pl.col("agent_id").alias("parent_agent_id"),
        pl.col("lineage_id").alias("expected_lineage"),
        pl.col("genome_id").alias("expected_genome"),
        (pl.col("generation") + 1).alias("expected_generation"),
        pl.col("birth_step").alias("parent_birth_step"),
    )
    invalid_births = births.join(parent, on="parent_agent_id", how="left").filter(
        pl.col("expected_genome").is_null()
        | (pl.col("parent_genome_id") != pl.col("expected_genome"))
        | (pl.col("lineage_id") != pl.col("expected_lineage"))
        | (pl.col("generation") != pl.col("expected_generation"))
        | (pl.col("birth_step") < pl.col("parent_birth_step"))
        | (pl.col("generation") < 1)
    )
    if invalid_births.limit(1).collect().height:
        raise ValueError(errors)
    endpoints = pl.concat(
        [
            deaths.select(
                "agent_id",
                pl.when(pl.col("parent_id") == _TELEMETRY_NULL_PARENT)
                .then(None)
                .otherwise(pl.col("parent_id"))
                .alias("parent_agent_id"),
                "lineage_id",
                "genome_id",
                "generation",
                "death_step",
                pl.col("age").alias("final_age"),
                pl.col("energy").cast(pl.Float64).alias("final_energy"),
                pl.col("cause_code").alias("death_cause_code"),
                pl.col("phase_code").alias("death_phase_code"),
                pl.lit(False).alias("alive_at_end"),
                pl.lit(None, dtype=pl.Int64).alias("observed_birth_step"),
            ),
            survivors.select(
                *_IDENTITY,
                pl.lit(None, dtype=pl.Int64).alias("death_step"),
                "final_age",
                "final_energy",
                pl.lit(None, dtype=pl.Int32).alias("death_cause_code"),
                pl.lit(None, dtype=pl.Int32).alias("death_phase_code"),
                pl.lit(True).alias("alive_at_end"),
                pl.col("birth_step").alias("observed_birth_step"),
            ),
        ]
    ).collect()
    if (
        origin.height != origin["agent_id"].n_unique()
        or endpoints.height != endpoints["agent_id"].n_unique()
        or origin.height != endpoints.height
    ):
        raise ValueError(errors)
    counts = births.group_by("parent_agent_id").agg(
        pl.len().cast(pl.Int64).alias("direct_offspring_count")
    )
    exact = origin.lazy().join(endpoints.lazy(), on=_IDENTITY, nulls_equal=True)
    result = (
        exact.join(
            counts,
            left_on="agent_id",
            right_on="parent_agent_id",
            how="left",
        )
        .with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.col("direct_offspring_count").fill_null(0),
        )
        .collect()
    )
    if (
        result.height != origin.height
        or result.filter(
            pl.any_horizontal(
                pl.col(
                    "agent_id",
                    "lineage_id",
                    "genome_id",
                    "generation",
                    "birth_step",
                    "final_age",
                    "final_energy",
                ).is_null(),
            )
            | pl.any_horizontal(pl.col("agent_id", "lineage_id", "genome_id") < 0)
            | ~pl.col("final_energy").is_finite()
            | (pl.col("birth_step") < 0)
            | (pl.col("birth_step") > final_step)
            | (pl.col("generation") < 0)
            | (pl.col("final_age") < 0)
            | (pl.col("death_step") < pl.col("birth_step"))
            | (pl.col("death_step") >= final_step)
            | (pl.col("observed_birth_step") != pl.col("birth_step"))
            | (pl.col("parent_agent_id").is_null() != (pl.col("generation") == 0))
        ).height
    ):
        raise ValueError(errors)
    return result.select(AGENT_SCHEMA.names()).cast(AGENT_SCHEMA).sort("agent_id")


def build_lineage_summaries(
    agents: pl.DataFrame,
    births: pl.LazyFrame,
    final_step: int,
) -> pl.DataFrame:
    """A death at D ends a lineage at state step D + 1."""
    alive = agents.filter(pl.col("alive_at_end")).height
    founder = pl.col("parent_agent_id").is_null() & (pl.col("generation") == 0)
    result = (
        agents.lazy()
        .group_by("lineage_id")
        .agg(
            pl.col("run_id").first(),
            founder.sum().alias("founders"),
            pl.col("agent_id").filter(founder).first().alias("founder_agent_id"),
            pl.col("genome_id").filter(founder).first().alias("founder_genome_id"),
            pl.col("birth_step").filter(founder).first().alias("founder_birth_step"),
            pl.len().alias("total_agents"),
            pl.col("alive_at_end").sum().alias("alive_final"),
            pl.col("generation").max().alias("max_generation"),
            pl.col("death_step").max().alias("last_death"),
            pl.col("agent_id")
            .sort_by(
                ["direct_offspring_count", "agent_id"],
                descending=[True, False],
            )
            .first()
            .alias("reproductive_leader_agent_id"),
            pl.col("direct_offspring_count").max().alias("reproductive_leader_offspring"),
            pl.col("agent_id")
            .sort_by(
                ["final_age", "agent_id"],
                descending=[True, False],
            )
            .first()
            .alias("survival_leader_agent_id"),
            pl.col("final_age").max().alias("survival_leader_age"),
        )
        .join(
            births.sort("child_agent_id")
            .group_by("lineage_id")
            .agg(
                pl.len().alias("births_total"),
                pl.col("birth_step").max().alias("last_birth_step"),
                pl.col("mutation_selected_count").sum().alias("mutation_selected_total"),
                pl.col("mutation_effective_count").sum().alias("mutation_effective_total"),
                pl.col("mutation_sum_abs_delta")
                .cast(pl.Float64)
                .sum()
                .alias("mutation_abs_delta_total"),
            ),
            on="lineage_id",
            how="left",
        )
        .with_columns(
            pl.col(
                "births_total",
                "mutation_selected_total",
                "mutation_effective_total",
                "mutation_abs_delta_total",
            ).fill_null(0),
            (pl.col("total_agents").cast(pl.Int64) - 1).alias("total_descendants"),
            (pl.col("total_agents") - pl.col("alive_final")).alias("deaths_total"),
            pl.when(pl.col("alive_final") == 0).then("last_death").alias("extinction_step"),
            (pl.col("alive_final") / max(alive, 1)).alias("final_population_fraction"),
        )
        .with_columns(
            (
                (pl.col("extinction_step") + 1).fill_null(final_step) - pl.col("founder_birth_step")
            ).alias("duration_steps"),
        )
        .collect()
    )
    if result.filter(
        (pl.col("founders") != 1)
        | (pl.col("births_total") != pl.col("total_descendants"))
        | (pl.col("duration_steps") < 0)
    ).height:
        raise ValueError(
            "each lineage must contain exactly one founder and all recorded descendants"
        )
    return result.select(LINEAGE_SCHEMA.names()).cast(LINEAGE_SCHEMA).sort("lineage_id")


def select_final_representatives(survivors: pl.DataFrame) -> pl.DataFrame:
    return survivors.sort(
        ["lineage_id", "generation", "agent_id"],
        descending=[False, True, False],
    ).unique(subset="lineage_id", keep="first", maintain_order=True)
