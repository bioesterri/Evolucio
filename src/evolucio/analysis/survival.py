"""Descriptive lifespan observations and Kaplan-Meier estimates."""

from dataclasses import dataclass
from typing import cast

import polars as pl

from evolucio.persistence.loaders.run_data import RunData
from evolucio.persistence.local.manifest import RunStatus
from evolucio.persistence.validation.types import CheckStatus

from .summaries import AGENTS_PATH
from .types import AnalysisPolicy, AnalysisUnavailableError, positive_int
from .windows import AnalysisRun, count, validated


@dataclass(frozen=True, slots=True)
class Cohort:
    start_step: int
    is_founder: bool


def survival_observations(
    run: RunData | AnalysisRun, cohort_width_steps: int = 100
) -> pl.LazyFrame:
    positive_int(cohort_width_steps, "cohort_width_steps")
    data = validated(run)
    if not data.population_available:
        raise AnalysisUnavailableError("validated lifecycle origins are unavailable")
    source, end = data.run, data.coverage.end_step_exclusive
    if source.manifest.status is RunStatus.COMPLETED:
        if (
            not data.has_artifact(AGENTS_PATH)
            or source.summary is None
            or data.report.components["summaries"] is not CheckStatus.PASSED
            or end != source.manifest.final_step
        ):
            raise AnalysisUnavailableError("validated terminal agent summaries are unavailable")
        observations = source.scan_agent_summaries().select(
            "agent_id",
            "birth_step",
            pl.col("parent_agent_id").is_null().alias("is_founder"),
            pl.col("final_age").alias("observed_age"),
            (~pl.col("alive_at_end")).alias("event_observed"),
            pl.when(~pl.col("alive_at_end"))
            .then(pl.col("death_cause_code"))
            .alias("death_cause_code"),
        )
    else:
        founders = source.scan_founders().select(
            "agent_id", "birth_step", pl.lit(True).alias("is_founder")
        )
        births = source.scan_births().select(
            pl.col("child_agent_id").alias("agent_id"),
            "birth_step",
            pl.lit(False).alias("is_founder"),
        )
        deaths = source.scan_deaths().select(
            "agent_id",
            pl.col("age").alias("death_age"),
            pl.col("cause_code").alias("death_cause_code"),
        )
        observations = (
            pl.concat([founders, births])
            .join(deaths, on="agent_id", how="left")
            .with_columns(
                pl.col("death_age").is_not_null().alias("event_observed"),
                pl.coalesce(
                    pl.col("death_age"),
                    end - pl.col("birth_step") - (~pl.col("is_founder")).cast(pl.Int64),
                ).alias("observed_age"),
            )
            .drop("death_age")
        )
    return (
        observations.with_columns(
            pl.lit(source.manifest.run_id).alias("run_id"),
            pl.col("agent_id", "birth_step", "observed_age", "death_cause_code").cast(pl.Int64),
            (pl.col("birth_step").cast(pl.Int64) // cohort_width_steps * cohort_width_steps).alias(
                "cohort_start"
            ),
        )
        .with_columns((pl.col("cohort_start") + cohort_width_steps).alias("cohort_end_exclusive"))
        .sort("agent_id")
    )


def cohort_summary(run: RunData | AnalysisRun, cohort_width_steps: int = 100) -> pl.LazyFrame:
    data = validated(run)
    observations = survival_observations(data, cohort_width_steps)
    age = pl.col("observed_age")
    deaths = pl.col("event_observed")
    return (
        observations.group_by("cohort_start", "cohort_end_exclusive", "is_founder")
        .agg(
            count().alias("agents_total"),
            deaths.sum().cast(pl.Int64).alias("deaths_observed"),
            (~deaths).sum().cast(pl.Int64).alias("right_censored_count"),
            age.filter(deaths).mean().alias("observed_death_age_mean"),
            age.filter(deaths).median().alias("observed_death_age_median"),
            age.min().alias("minimum_observed_age"),
            age.max().alias("maximum_observed_age"),
            pl.col("birth_step").min().alias("first_birth_step"),
            pl.col("birth_step").max().alias("last_birth_step"),
        )
        .with_columns(
            pl.lit(data.coverage.end_step_exclusive)
            .cast(pl.Int64)
            .alias("observation_end_state_step")
        )
        .sort("cohort_start", "is_founder")
    )


def kaplan_meier(observations: pl.LazyFrame) -> pl.LazyFrame:
    """Each tied death and censoring contributes to the risk set before removal."""
    observed = pl.col("event_observed")
    return (
        observations.group_by(pl.col("observed_age").alias("age"))
        .agg(
            count().alias("exits"),
            observed.sum().cast(pl.Int64).alias("death_count"),
            (~observed).sum().cast(pl.Int64).alias("censored_count"),
        )
        .sort("age")
        .with_columns(pl.col("exits").cum_sum(reverse=True).alias("n_at_risk"))
        .with_columns(
            (1 - pl.col("death_count") / pl.col("n_at_risk"))
            .cum_prod()
            .alias("survival_probability")
        )
        .drop("exits")
    )


def survival_curve(
    run: RunData | AnalysisRun,
    cohort: Cohort | None = None,
    *,
    cohort_width_steps: int = 100,
) -> pl.LazyFrame:
    observations = survival_observations(run, cohort_width_steps)
    if cohort is not None:
        if (
            type(cohort.start_step) is not int
            or cohort.start_step < 0
            or cohort.start_step % cohort_width_steps
            or type(cohort.is_founder) is not bool
        ):
            raise ValueError("cohort must identify an absolute aligned start and founder status")
        observations = observations.filter(
            (pl.col("cohort_start") == cohort.start_step)
            & (pl.col("is_founder") == cohort.is_founder)
        )
    return kaplan_meier(observations)


def median_survival_age(curve: pl.LazyFrame) -> int | None:
    return cast(
        int | None,
        curve.filter(pl.col("survival_probability") <= 0.5)
        .select(pl.col("age").min())
        .collect()
        .item(),
    )


@dataclass(frozen=True, slots=True)
class SurvivalSummary:
    agents_total: int
    deaths_observed: int
    right_censored_count: int
    median_survival_age: int | None


def survival_summary(run: AnalysisRun, policy: AnalysisPolicy) -> SurvivalSummary:
    totals = (
        survival_observations(run, policy.cohort_width_steps)
        .select(
            count().alias("agents_total"),
            pl.col("event_observed").sum().cast(pl.Int64).alias("deaths"),
        )
        .collect()
        .row(0, named=True)
    )
    return SurvivalSummary(
        totals["agents_total"],
        totals["deaths"],
        totals["agents_total"] - totals["deaths"],
        median_survival_age(survival_curve(run, cohort_width_steps=policy.cohort_width_steps)),
    )
