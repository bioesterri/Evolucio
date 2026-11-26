"""Aggregate persisted mutation effects without inferring genome contents."""

import polars as pl

from evolucio.core.policy.schema import POLICY_PARAMETER_COUNT
from evolucio.persistence.loaders.run_data import RunData

from .genealogy import genealogy_data, get_ancestors
from .types import AnalysisWindow
from .windows import AnalysisRun, count, ratio, window_grid, window_key, within


def _aggregate(births: pl.LazyFrame, key: str | pl.Expr) -> pl.LazyFrame:
    selected = pl.col("mutation_selected_count")
    effective = pl.col("mutation_effective_count")
    delta = pl.col("mutation_sum_abs_delta").cast(pl.Float64)
    return births.group_by(key).agg(
        count().alias("births_total"),
        selected.sum().alias("mutation_selected_total"),
        effective.sum().alias("mutation_effective_total"),
        pl.col("mutation_selected_weight_count").sum().alias("selected_weights"),
        pl.col("mutation_selected_bias_count").sum().alias("selected_biases"),
        (selected > 0).sum().alias("births_with_selected_mutations"),
        (effective > 0).sum().alias("births_with_effective_mutations"),
        delta.sum().alias("mutation_abs_delta_total"),
        pl.col("mutation_max_abs_delta").max().alias("maximum_abs_delta"),
        effective.mean().alias("mean_effective_mutations_per_birth"),
        delta.mean().alias("mean_abs_delta_per_birth"),
    )


def _complete(frame: pl.LazyFrame) -> pl.LazyFrame:
    return (
        frame.with_columns(
            pl.col(
                "births_total",
                "mutation_selected_total",
                "mutation_effective_total",
                "births_with_selected_mutations",
                "births_with_effective_mutations",
                "mutation_abs_delta_total",
                "selected_weights",
                "selected_biases",
            ).fill_null(0),
        )
        .with_columns(
            (pl.col("births_total") * POLICY_PARAMETER_COUNT).alias(
                "total_parameter_opportunities"
            ),
            pl.col("mutation_selected_total").alias("selected_parameters"),
            pl.col("mutation_effective_total").alias("effective_parameters"),
        )
        .with_columns(
            ratio(pl.col("effective_parameters"), pl.col("total_parameter_opportunities")).alias(
                "effective_mutation_fraction"
            )
        )
    )


def mutation_summary_by_lineage(
    run: RunData | AnalysisRun, window: AnalysisWindow | None = None
) -> pl.LazyFrame:
    data = genealogy_data(run)
    births = data.run.scan_births().filter(within("birth_step", data.bounds(window)))
    return _complete(
        data.run.scan_founders()
        .select("lineage_id")
        .join(
            _aggregate(births, "lineage_id"),
            on="lineage_id",
            how="left",
        )
    ).sort("lineage_id")


def mutation_summary_by_generation(run: RunData | AnalysisRun) -> pl.LazyFrame:
    data = genealogy_data(run)
    births = data.run.scan_births()
    generations = pl.concat(
        [data.run.scan_founders().select("generation"), births.select("generation")],
        how="vertical_relaxed",
    ).unique()
    return _complete(
        generations.join(_aggregate(births, "generation"), on="generation", how="left")
    ).sort("generation")


def mutation_summary_by_window(
    run: RunData | AnalysisRun, window_steps: int, window: AnalysisWindow | None = None
) -> pl.LazyFrame:
    data = genealogy_data(run)
    coverage = data.bounds(window)
    births = data.run.scan_births().filter(within("birth_step", coverage))
    return _complete(
        window_grid(window_steps, coverage).join(
            _aggregate(births, window_key("birth_step", window_steps)),
            on="window_start",
            how="left",
        )
    ).sort("window_start")


def ancestral_mutation_totals(run: RunData | AnalysisRun, agent_id: int) -> dict[str, object]:
    data = genealogy_data(run)
    ancestors = get_ancestors(data, agent_id)["agent_id"].to_list()
    return (
        data.run.scan_genealogy()
        .filter(pl.col("child_agent_id").is_in([agent_id, *ancestors]))
        .select(
            pl.col("mutation_effective_count").sum().alias("path_effective_mutations"),
            pl.col("mutation_sum_abs_delta").cast(pl.Float64).sum().alias("path_sum_abs_delta"),
        )
        .collect()
        .row(0, named=True)
    )
