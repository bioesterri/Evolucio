"""Lineage proportions and diversity on a sparse event timeline."""

from collections import Counter
from heapq import heappop, heappush
from itertools import groupby
from math import exp, log

import polars as pl

from evolucio.persistence.loaders.run_data import RunData

from .genealogy import genealogy_data
from .lineage_dynamics import lineage_events, lineage_population_series
from .types import AnalysisWindow
from .windows import AnalysisRun, ratio, within

DIVERSITY_SCHEMA = {
    "step": pl.Int64,
    "population": pl.Int64,
    "active_lineages": pl.Int64,
    "dominant_lineage_id": pl.Int64,
    "dominant_lineage_fraction": pl.Float64,
    "dominant_lineage_tie_count": pl.Int64,
    "shannon_entropy": pl.Float64,
    "simpson_concentration": pl.Float64,
    "simpson_diversity": pl.Float64,
    "effective_lineages_shannon": pl.Float64,
    "effective_lineages_simpson": pl.Float64,
    "lineage_evenness": pl.Float64,
}


def _diversity(events: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    bounds = pl.DataFrame(
        {"step": [start], "lineage_id": [None], "population_after": [None]}, schema=events.schema
    )
    ordered = pl.concat([events, bounds]).filter(pl.col("step") < end).sort("step")
    sizes: dict[int, int] = {}
    multiplicity: Counter[int] = Counter()
    heap: list[tuple[int, int]] = []
    population = squares = 0
    log_sum = 0.0
    rows: list[tuple[object, ...]] = []
    for step, changes in groupby(ordered.iter_rows(), key=lambda row: row[0]):
        for _, lineage, after in changes:
            if lineage is None:
                continue
            before = sizes.pop(lineage, 0)
            if before == after:
                if before:
                    sizes[lineage] = before
                continue
            population += after - before
            squares += after**2 - before**2
            if before:
                log_sum -= before * log(before)
                multiplicity[before] -= 1
            if after:
                sizes[lineage] = after
                log_sum += after * log(after)
                multiplicity[after] += 1
                heappush(heap, (-after, lineage))
        if step < start:
            continue
        if population == 0:
            rows.append((step, 0, 0, None, None, 0, None, None, None, None, None, None))
            log_sum = 0.0
            continue
        while heap and sizes.get(heap[0][1]) != -heap[0][0]:
            heappop(heap)
        neg_maximum, dominant = heap[0]
        richness = len(sizes)
        entropy = 0.0 if richness == 1 else max(0.0, log(population) - log_sum / population)
        concentration = squares / population**2
        rows.append(
            (
                step,
                population,
                richness,
                dominant,
                -neg_maximum / population,
                multiplicity[-neg_maximum],
                entropy,
                concentration,
                1 - concentration,
                exp(entropy),
                1 / concentration,
                entropy / log(richness) if richness > 1 else None,
            )
        )
    return pl.DataFrame(rows, schema=DIVERSITY_SCHEMA, orient="row")


def lineage_diversity_series(
    run: RunData | AnalysisRun, window: AnalysisWindow | None = None
) -> pl.LazyFrame:
    data = genealogy_data(run)
    coverage = data.bounds(window)
    events = lineage_events(data).select("step", "lineage_id", "population_after").collect()
    series = (
        _diversity(events, data.coverage.start_step, coverage.end_step_exclusive)
        .lazy()
        .sort("step")
    )
    metrics = (
        data.run.scan_metrics()
        .filter(
            pl.col("step").is_between(
                data.coverage.start_step, coverage.end_step_exclusive, closed="left"
            )
        )
        .select("step", "alive_count", "active_lineage_count")
        .sort("step")
    )
    check = metrics.join_asof(series, on="step").filter(
        (pl.col("alive_count") != pl.col("population"))
        | (pl.col("active_lineage_count") != pl.col("active_lineages"))
    )
    if check.limit(1).collect().height:
        raise ValueError("lineage diversity disagrees with persisted population metrics")
    start = coverage.start_step
    anchor = (
        series.filter(pl.col("step") <= start)
        .tail(1)
        .with_columns(pl.lit(start, dtype=pl.Int64).alias("step"))
    )
    series = pl.concat([anchor, series.filter(within("step", coverage) & (pl.col("step") > start))])
    return series.with_columns(
        pl.col("step").shift(-1).fill_null(coverage.end_step_exclusive).alias("end_step_exclusive"),
        pl.lit(data.run.manifest.run_id).alias("run_id"),
    )


def lineage_proportions_at_step(run: RunData | AnalysisRun, step: int) -> pl.LazyFrame:
    frame = lineage_population_series(run, window=AnalysisWindow(step, step + 1))
    return frame.filter(pl.col("population_after") > 0).with_columns(
        ratio(pl.col("population_after"), pl.col("population_after").sum()).alias(
            "population_fraction"
        )
    )
