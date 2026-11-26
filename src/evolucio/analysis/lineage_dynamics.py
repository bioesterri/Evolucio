"""Exact lineage demography using event rows and constant-population intervals."""

from collections.abc import Sequence

import polars as pl

from evolucio.persistence.loaders.run_data import RunData
from evolucio.persistence.validation.types import CheckStatus

from .genealogy import genealogy_data, genealogy_nodes
from .offspring import reproductive_success_by_agent
from .population import population_changes
from .summaries import LINEAGES_PATH
from .survival import kaplan_meier
from .types import AnalysisWindow, positive_int
from .windows import AnalysisRun, count, ratio, window_bounds, within

COUNTS = ("birth_count", "death_count")


def lineage_events(data: AnalysisRun) -> pl.LazyFrame:
    """Return all lineage changes and census anchors, with an exact balance check."""
    start = data.coverage.start_step
    source = data.run
    founders = source.scan_founders().group_by("lineage_id").agg(count().alias("initial"))
    births = (
        source.scan_births()
        .select("lineage_id", pl.col("birth_step").alias("step"))
        .with_columns(
            pl.lit(1, dtype=pl.Int64).alias("birth_count"),
            pl.lit(0, dtype=pl.Int64).alias("death_count"),
        )
    )
    deaths = (
        source.scan_deaths()
        .select("lineage_id", pl.col("death_step").alias("step"))
        .with_columns(
            pl.lit(0, dtype=pl.Int64).alias("birth_count"),
            pl.lit(1, dtype=pl.Int64).alias("death_count"),
        )
    )
    anchors = founders.select(
        "lineage_id",
        pl.lit(start, dtype=pl.Int64).alias("step"),
        *(pl.lit(0, dtype=pl.Int64).alias(name) for name in COUNTS),
    )
    events = (
        pl.concat([anchors, births, deaths], how="vertical_relaxed")
        .group_by("lineage_id", "step")
        .agg(pl.col(*COUNTS).sum())
        .join(founders, on="lineage_id", how="left")
        .sort("lineage_id", "step")
        .with_columns((pl.col("birth_count") - pl.col("death_count")).alias("net_change"))
        .with_columns(
            (pl.col("initial") + pl.col("net_change").cum_sum().over("lineage_id")).alias(
                "population_after"
            )
        )
        .with_columns(
            (pl.col("population_after") - pl.col("net_change")).alias("population_before"),
            pl.lit(source.manifest.run_id).alias("run_id"),
        )
        .drop("initial")
        .collect()
    )
    changes, initial = population_changes(data)
    totals = (
        events.lazy()
        .group_by("step")
        .agg(pl.col(*COUNTS).sum())
        .join(changes, on="step", how="full", coalesce=True, suffix="_global")
        .with_columns(
            pl.col(
                "birth_count", "death_count", "birth_count_global", "death_count_global"
            ).fill_null(0)
        )
    )
    if (
        events.filter(pl.col("population_after") < 0).height
        or events.filter(pl.col("step") == start)["population_before"].sum() != initial
        or totals.filter(
            (pl.col("birth_count") != pl.col("birth_count_global"))
            | (pl.col("death_count") != pl.col("death_count_global"))
        )
        .limit(1)
        .collect()
        .height
    ):
        raise ValueError("lineage population balance differs from PR-43")
    return events.lazy()


def lineage_population_series(
    run: RunData | AnalysisRun,
    lineage_ids: Sequence[int] | None = None,
    window: AnalysisWindow | None = None,
    *,
    dense: bool = False,
    max_rows: int = 100_000,
) -> pl.LazyFrame:
    data = genealogy_data(run)
    coverage = data.bounds(window)
    start, end = coverage.start_step, coverage.end_step_exclusive
    events = lineage_events(data)
    if lineage_ids is not None:
        events = events.filter(pl.col("lineage_id").is_in(lineage_ids))
    # Add the pre-window baseline even for lineages with no events inside the window.
    prior = (
        events.filter(pl.col("step") < start)
        .group_by("run_id", "lineage_id")
        .agg(pl.col("population_after").last())
        .with_columns(
            pl.lit(start, dtype=pl.Int64).alias("step"),
            pl.col("population_after").alias("population_before"),
            *(pl.lit(0, dtype=pl.Int64).alias(name) for name in (*COUNTS, "net_change")),
        )
    )
    selected = events.filter(within("step", coverage))
    prior = prior.join(
        selected.filter(pl.col("step") == start).select("lineage_id"), on="lineage_id", how="anti"
    ).filter(pl.col("step") < end)
    series = pl.concat([selected, prior.select(selected.collect_schema().names())]).sort(
        "lineage_id", "step"
    )
    series = series.with_columns(
        pl.col("step").shift(-1).over("lineage_id").fill_null(end).alias("end_step_exclusive")
    )
    if not dense:
        return series
    positive_int(max_rows, "max_rows")
    if (
        int(series.select((pl.col("end_step_exclusive") - pl.col("step")).sum()).collect().item())
        > max_rows
    ):
        raise ValueError("dense lineage series exceeds max_rows")
    return (
        series.with_columns(pl.int_ranges("step", "end_step_exclusive").alias("dense_step"))
        .explode("dense_step", empty_as_null=False)
        .with_columns(
            *(
                pl.when(pl.col("dense_step") == pl.col("step")).then(name).otherwise(0).alias(name)
                for name in (*COUNTS, "net_change")
            ),
            pl.when(pl.col("dense_step") == pl.col("step"))
            .then("population_before")
            .otherwise("population_after")
            .alias("population_before"),
        )
        .drop("step", "end_step_exclusive")
        .rename({"dense_step": "step"})
    )


def lineage_extinction_summary(run: RunData | AnalysisRun) -> pl.LazyFrame:
    data = genealogy_data(run)
    end = data.coverage.end_step_exclusive
    nodes = genealogy_nodes(data)
    maxima = (
        lineage_events(data)
        .group_by("lineage_id")
        .agg(
            pl.max_horizontal(
                pl.col("population_before").max(), pl.col("population_after").max()
            ).alias("maximum_population")
        )
    )
    if (
        data.report.components["summaries"] is CheckStatus.PASSED
        and data.has_artifact(LINEAGES_PATH)
        and end == data.run.manifest.final_step
    ):
        base = data.run.scan_lineage_summaries().select(
            "run_id",
            "lineage_id",
            "founder_agent_id",
            "founder_birth_step",
            "total_agents",
            "total_descendants",
            "alive_final",
            pl.col("extinction_step").alias("extinction_event_step"),
        )
    else:
        founder = pl.col("parent_agent_id").is_null()
        base = (
            nodes.group_by("run_id", "lineage_id")
            .agg(
                pl.col("agent_id").filter(founder).first().alias("founder_agent_id"),
                pl.col("birth_step").filter(founder).first().alias("founder_birth_step"),
                count().alias("total_agents"),
                (count() - 1).alias("total_descendants"),
                pl.col("death_step").null_count().cast(pl.Int64).alias("alive_final"),
                pl.col("death_step").max().alias("last_death"),
            )
            .with_columns(
                pl.when(pl.col("alive_final") == 0)
                .then("last_death")
                .alias("extinction_event_step")
            )
            .drop("last_death")
        )
    return (
        base.join(maxima, on="lineage_id")
        .with_columns(
            (pl.col("extinction_event_step") + 1).alias("extinction_state_step"),
            pl.col("extinction_event_step").is_not_null().alias("extinction_observed"),
        )
        .with_columns(
            (pl.col("extinction_state_step").fill_null(end) - pl.col("founder_birth_step")).alias(
                "duration_observed_steps"
            )
        )
        .sort("lineage_id")
    )


def lineage_population_by_window(
    run: RunData | AnalysisRun,
    window_steps: int,
    window: AnalysisWindow | None = None,
    *,
    lineage_ids: Sequence[int] | None = None,
) -> pl.LazyFrame:
    positive_int(window_steps, "window_steps")
    data = genealogy_data(run)
    coverage = data.bounds(window)
    step = pl.col("step")
    left = pl.col("window_start")
    right = pl.col("end_step_exclusive")
    after = pl.col("population_after")
    begins = step >= left
    span = pl.min_horizontal(right, left + window_steps) - pl.max_horizontal(step, left)
    return (
        lineage_population_series(data, lineage_ids, window)
        .with_columns(
            pl.int_ranges(step // window_steps * window_steps, right, window_steps).alias(
                "window_start"
            )
        )
        .explode("window_start", empty_as_null=False)
        .group_by("run_id", "lineage_id", "window_start")
        .agg(
            pl.when(begins)
            .then("population_before")
            .otherwise(after)
            .first()
            .alias("population_start"),
            after.last().alias("population_end"),
            after.min().alias("population_min"),
            after.max().alias("population_max"),
            ((after * span).sum() / span.sum()).alias("population_mean"),
            pl.col("birth_count").filter(begins).sum().alias("births_total"),
            pl.col("death_count").filter(begins).sum().alias("deaths_total"),
            (after * span - pl.when(begins).then("net_change").otherwise(0))
            .sum()
            .alias("agent_steps_at_start"),
            span.sum().alias("steps_covered"),
        )
        .with_columns(
            *window_bounds(window_steps, coverage),
            (pl.col("steps_covered") < window_steps).alias("window_incomplete"),
            (pl.col("births_total") - pl.col("deaths_total")).alias("net_change"),
            *(
                ratio(pl.col(f"{event}_total"), pl.col("agent_steps_at_start")).alias(
                    f"{event}_per_agent_step"
                )
                for event in ("births", "deaths")
            ),
        )
        .with_columns(
            pl.when(pl.col("net_change") > 0)
            .then(pl.lit("expansion"))
            .when(pl.col("net_change") < 0)
            .then(pl.lit("decline"))
            .otherwise(pl.lit("zero_net_change"))
            .alias("population_trend")
        )
        .sort("lineage_id", "window_start")
    )


def survival_curve_by_lineage(
    run: RunData | AnalysisRun,
    *,
    lineage_ids: Sequence[int],
) -> pl.LazyFrame:
    agents = reproductive_success_by_agent(run).select(
        "lineage_id", "observed_age", (~pl.col("alive_at_end")).alias("event_observed")
    )
    curves = [
        kaplan_meier(agents.filter(pl.col("lineage_id") == lineage_id)).with_columns(
            pl.lit(lineage_id, dtype=pl.Int64).alias("lineage_id")
        )
        for lineage_id in sorted(set(lineage_ids))
    ]
    if not curves:
        raise ValueError("provide at least one lineage_id")
    return pl.concat(curves)
