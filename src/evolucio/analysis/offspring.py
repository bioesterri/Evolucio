"""Observed reproduction and follow-up over the verified lifecycle prefix."""

from dataclasses import dataclass

import polars as pl

from evolucio.persistence.loaders.run_data import RunData
from evolucio.persistence.validation.types import CheckStatus

from .genealogy import genealogy_data, genealogy_nodes
from .summaries import AGENTS_PATH
from .survival import entry_state_step
from .types import positive_int
from .windows import AnalysisRun, count, ratio


def reproductive_success_by_agent(run: RunData | AnalysisRun) -> pl.LazyFrame:
    data = genealogy_data(run)
    source = data.run
    end = data.coverage.end_step_exclusive
    nodes = genealogy_nodes(data)
    children = (
        source.scan_births()
        .group_by("parent_agent_id")
        .agg(
            count().alias("direct_offspring_count"),
            pl.col("birth_step").min().alias("first_reproduction_step"),
        )
    )
    result = nodes.join(
        children, left_on="agent_id", right_on="parent_agent_id", how="left"
    ).with_columns(
        pl.col("direct_offspring_count").fill_null(0),
        pl.col("death_step").is_null().alias("alive_at_end"),
        entry_state_step(pl.col("parent_agent_id").is_null()).alias("entry_state_step"),
    )
    if (
        data.has_artifact(AGENTS_PATH)
        and data.report.components["summaries"] is CheckStatus.PASSED
        and end == source.manifest.final_step
    ):
        summary = source.scan_agent_summaries().select(
            "agent_id",
            pl.col("direct_offspring_count").alias("summary_offspring_count"),
            pl.col("final_age").alias("observed_age"),
        )
        result = result.join(summary, on="agent_id", how="left")
        mismatch = result.filter(
            pl.col("direct_offspring_count") != pl.col("summary_offspring_count").fill_null(-1)
        )
        if mismatch.limit(1).collect().height:
            raise ValueError("agent summaries and births disagree on direct offspring")
        result = result.drop("summary_offspring_count")
    else:
        result = (
            result.join(source.scan_deaths().select("agent_id", "age"), on="agent_id", how="left")
            .with_columns(
                pl.coalesce("age", end - pl.col("entry_state_step")).alias("observed_age")
            )
            .drop("age")
        )
    return (
        result.with_columns(
            (pl.col("direct_offspring_count") > 0).alias("reproduced"),
            (pl.col("first_reproduction_step") - pl.col("entry_state_step")).alias(
                "time_to_first_reproduction"
            ),
            (end - pl.col("entry_state_step")).alias("follow_up_steps"),
        )
        .with_columns(
            (~pl.col("reproduced") & pl.col("alive_at_end")).alias("reproduction_right_censored"),
            (~pl.col("reproduced") & ~pl.col("alive_at_end")).alias("died_without_offspring"),
        )
        .sort("agent_id")
    )


@dataclass(frozen=True, slots=True)
class OffspringDistribution:
    summary: dict[str, object]
    histogram: pl.LazyFrame


def offspring_distribution(run: RunData | AnalysisRun) -> OffspringDistribution:
    agents = reproductive_success_by_agent(run)
    offspring = pl.col("direct_offspring_count")
    summary = (
        agents.select(
            count().alias("agents_total"),
            pl.col("reproduced").sum().alias("agents_with_offspring"),
            (~pl.col("reproduced")).sum().alias("agents_without_offspring_observed"),
            pl.col("died_without_offspring").sum().alias("agents_dead_without_offspring"),
            pl.col("reproduction_right_censored")
            .sum()
            .alias("agents_alive_without_offspring_observed"),
            offspring.mean().alias("mean_direct_offspring"),
            offspring.median().alias("median_direct_offspring"),
            offspring.max().alias("maximum_direct_offspring"),
            offspring.std(ddof=0).alias("offspring_std"),
            (~pl.col("reproduced")).mean().alias("proportion_without_offspring_observed"),
        )
        .collect()
        .row(0, named=True)
    )
    histogram = (
        agents.group_by(offspring.alias("offspring_count"))
        .agg(count().alias("agent_count"))
        .with_columns(
            ratio(pl.col("agent_count"), pl.col("agent_count").sum()).alias("agent_fraction")
        )
        .sort("offspring_count")
    )
    return OffspringDistribution(summary, histogram)


def generation_summary(
    run: RunData | AnalysisRun, *, follow_up_steps: int | None = None
) -> pl.LazyFrame:
    data = genealogy_data(run)
    agents = reproductive_success_by_agent(data)
    births = data.run.scan_births()
    if follow_up_steps is not None:
        positive_int(follow_up_steps, "follow_up_steps")
        agents = agents.filter(pl.col("follow_up_steps") >= follow_up_steps)
        follow_up_births = births.join(
            agents.select("agent_id", "entry_state_step"),
            left_on="parent_agent_id",
            right_on="agent_id",
        ).filter(pl.col("birth_step") < pl.col("entry_state_step") + follow_up_steps)
        agents = (
            agents.drop("direct_offspring_count")
            .join(
                follow_up_births.group_by("parent_agent_id").agg(
                    count().alias("direct_offspring_count")
                ),
                left_on="agent_id",
                right_on="parent_agent_id",
                how="left",
            )
            .with_columns(pl.col("direct_offspring_count").fill_null(0))
        )
    dead = ~pl.col("alive_at_end")
    age = pl.col("observed_age")
    offspring = pl.col("direct_offspring_count")
    mutations = births.select(
        "child_agent_id", "mutation_selected_count", "mutation_effective_count"
    )
    return (
        agents.join(mutations, left_on="agent_id", right_on="child_agent_id", how="left")
        .group_by("run_id", "generation")
        .agg(
            count().alias("agents_born"),
            dead.sum().alias("agents_dead"),
            pl.col("alive_at_end").sum().alias("agents_alive_final"),
            pl.col("alive_at_end").mean().alias("proportion_alive_final"),
            offspring.mean().alias("mean_direct_offspring"),
            offspring.median().alias("median_direct_offspring"),
            age.mean().alias("observed_age_mean"),
            age.filter(dead).mean().alias("observed_death_age_mean"),
            age.filter(dead).median().alias("observed_death_age_median"),
            pl.col("follow_up_steps").mean().alias("follow_up_steps_mean"),
            pl.col("follow_up_steps").min().alias("follow_up_steps_min"),
            pl.col("mutation_selected_count").sum().alias("mutation_selected_total"),
            pl.col("mutation_effective_count").sum().alias("mutation_effective_total"),
        )
        .with_columns(
            pl.lit(follow_up_steps, dtype=pl.Int64).alias("common_offspring_follow_up_steps")
        )
        .sort("generation")
    )
