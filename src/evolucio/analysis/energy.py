"""Stocks, complete flows and a signed live-energy budget."""

from dataclasses import dataclass
from math import isfinite
from typing import cast

import numpy as np
import polars as pl

from evolucio.persistence.loaders.run_data import RunData
from evolucio.persistence.local.artifacts import artifact_path
from evolucio.persistence.validation.types import CheckStatus
from evolucio.persistence.zarr.checkpoint import read_checkpoint_arrays

from .population import PopulationSummary, population_by_window
from .summaries import AGENTS_PATH
from .types import AnalysisWindow, Coverage, Metric, Quality
from .windows import AnalysisRun, count, ratio, validated, window_grid, window_key, within

COSTS = (
    "basal_energy_cost",
    "movement_energy_cost",
    "feeding_energy_cost",
    "reproduction_energy_cost",
)
FLOWS = ("feeding_energy_gained", *COSTS, "resource_consumed")
STOCKS = ("total_energy_alive", "mean_energy_alive", "total_resources")


def energy_series(run: RunData | AnalysisRun, window: AnalysisWindow | None = None) -> pl.LazyFrame:
    data = validated(run)
    return (
        data.run.scan_metrics()
        .filter(within("step", data.bounds(window)))
        .select("run_id", pl.col("step").cast(pl.Int64), pl.col(*STOCKS, *FLOWS).cast(pl.Float64))
        .with_columns(
            _gross("").alias("gross_energy_cost"),
            pl.col("total_resources").alias("total_resources_available"),
            pl.lit(data.metrics_interval).cast(pl.Int64).alias("metrics_interval"),
        )
    )


def _gross(suffix: str) -> pl.Expr:
    return pl.sum_horizontal(*(pl.col(f"{name}{suffix}") for name in COSTS), ignore_nulls=False)


def _finite_sum(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.Float64)
    return pl.when(value.is_finite().all()).then(value.sum())


def _event_energy(run: AnalysisRun, coverage: Coverage) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    births = run.run.scan_births().filter(within("birth_step", coverage))
    deaths = run.run.scan_deaths().filter(within("death_step", coverage))
    return births, deaths


def energy_by_window(
    run: RunData | AnalysisRun, window_steps: int, window: AnalysisWindow | None = None
) -> pl.LazyFrame:
    """Aggregate observations and label complete versus unavailable flow totals."""
    data = validated(run)
    coverage = data.bounds(window)
    series = energy_series(data, window)
    births, deaths = _event_energy(data, coverage)
    windows = window_grid(window_steps, coverage)
    if data.population_available:
        exposure = population_by_window(data, window_steps, window).select(
            "window_start", "agent_steps_at_start"
        )
        windows = windows.join(exposure, on="window_start", how="left")
    else:
        windows = windows.with_columns(pl.lit(None, dtype=pl.Int64).alias("agent_steps_at_start"))
    start = coverage.start_step
    first_end = min((start // window_steps + 1) * window_steps, coverage.end_step_exclusive)
    capture = (
        _checkpoint(data, first_end)
        if start == data.coverage.start_step and start < first_end
        else None
    )
    accumulator = capture[2] if capture else {}
    samples = series.group_by(window_key("step", window_steps)).agg(
        count().alias("metric_samples"),
        *(_finite_sum(name).alias(f"{name}_total") for name in FLOWS),
        pl.col("total_resources").min().alias("observed_min_resources"),
        pl.col("total_resources").max().alias("observed_max_resources"),
        pl.col("total_resources").mean().alias("observed_mean_resources"),
        pl.col("total_energy_alive").mean().alias("observed_mean_energy_alive"),
    )
    newborns = births.group_by(window_key("birth_step", window_steps)).agg(
        count().alias("births_total"),
        _finite_sum("child_initial_energy").alias("offspring_energy_total"),
    )
    removed = deaths.group_by(window_key("death_step", window_steps)).agg(
        count().alias("deaths_total"),
        _finite_sum("energy").alias("energy_at_death_total"),
        _positive_death_energy().alias("positive_energy_removed_on_death"),
    )
    complete = pl.col("metric_samples") == pl.col("steps_covered")
    return (
        windows.join(samples, on="window_start", how="left")
        .join(newborns, on="window_start", how="left")
        .join(removed, on="window_start", how="left")
        .with_columns(pl.col("metric_samples", "births_total", "deaths_total").fill_null(0))
        .with_columns(
            pl.when(complete)
            .then(pl.lit(Quality.EXACT.value))
            .when(pl.col("metric_samples") > 0)
            .then(pl.lit(Quality.SAMPLED.value))
            .otherwise(pl.lit(Quality.UNAVAILABLE.value))
            .alias("stock_statistics_quality"),
            *(
                pl.when(complete)
                .then(pl.col(f"{name}_total"))
                .when(pl.col("covered_start") == start)
                .then(pl.lit(_number(accumulator.get(name)), dtype=pl.Float64))
                .alias(f"{name}_total")
                for name in FLOWS
            ),
            pl.when(pl.col("births_total") == 0)
            .then(0.0)
            .otherwise(pl.col("offspring_energy_total"))
            .alias("offspring_energy_total"),
            *(
                pl.when(pl.col("deaths_total") == 0).then(0.0).otherwise(pl.col(name)).alias(name)
                for name in ("energy_at_death_total", "positive_energy_removed_on_death")
            ),
        )
        .with_columns(
            _gross("_total").alias("gross_energy_cost_total"),
            (pl.col("reproduction_energy_cost_total") - pl.col("offspring_energy_total")).alias(
                "net_reproductive_energy_cost_total"
            ),
        )
        .with_columns(
            *(
                ratio(pl.col(numerator), pl.col(denominator)).alias(name)
                for name, numerator, denominator in EFFICIENCY
            ),
        )
        .sort("window_start")
    )


EFFICIENCY = (
    ("births_per_resource_consumed", "births_total", "resource_consumed_total"),
    ("feeding_energy_per_resource", "feeding_energy_gained_total", "resource_consumed_total"),
    ("reproduction_cost_per_birth", "reproduction_energy_cost_total", "births_total"),
    ("agent_steps_per_energy_cost", "agent_steps_at_start", "gross_energy_cost_total"),
)


def _positive_death_energy() -> pl.Expr:
    energy = pl.col("energy").cast(pl.Float64)
    return pl.when(energy.is_finite().all()).then(energy.clip(lower_bound=0).sum())


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and isfinite(value) else None


def _checkpoint(run: AnalysisRun, step: int) -> tuple[float, float, dict[str, float], bool] | None:
    for record in run.run.list_checkpoints():
        if record.state_step != step:
            continue
        _, arrays = read_checkpoint_arrays(artifact_path(run.run.path, record.relative_path))
        saturated = bool(arrays["accumulator/count_overflow"].item())
        totals: dict[str, float] = {}
        if (
            not saturated
            and record.committed_telemetry_end_step == step
            and record.telemetry_prefix_digest is not None
            and arrays["accumulator/steps_accumulated"].item() == step - run.coverage.start_step
        ):
            totals = {name: float(arrays[f"accumulator/{name}_total"].item()) for name in FLOWS}
        return (
            float(arrays["population/energy"][arrays["population/alive"]].sum()),
            float(arrays["world/resources"].sum()),
            totals,
            saturated,
        )
    return None


def _stocks(
    run: AnalysisRun,
    step: int,
    checkpoint: tuple[float, float, dict[str, float], bool] | None,
) -> tuple[tuple[float | None, str], tuple[float | None, str]]:
    if checkpoint is not None:
        return (_number(checkpoint[0]), "checkpoint"), (_number(checkpoint[1]), "checkpoint")
    if any(record.state_step == step for record in run.run.list_snapshots()):
        snapshot = run.run.load_snapshot(step)
        return (
            (_number(float(snapshot.population.energy.sum())), "snapshot"),
            (_number(float(snapshot.world.resources.sum())), "snapshot"),
        )
    row = (
        run.run.scan_metrics()
        .filter(pl.col("step") == step - 1)
        .select("total_energy_alive", "total_resources")
        .collect()
    )
    if row.height:
        return (_number(row.item(0, 0)), "metrics"), (_number(row.item(0, 1)), "metrics")
    energy: float | None = None
    resources: float | None = None
    energy_source = resource_source = "state observation"
    if step == 0:
        if run.population_available:
            energy = _number(
                run.run.scan_founders().select(_finite_sum("initial_energy")).collect().item()
            )
            energy_source = "founders"
        world = run.run.config.world
        if world.resource_distribution == "uniform" or world.resource_patch_contrast == 0:
            resources = world.width * world.height * float(np.float32(world.initial_resource_mean))
            resource_source = "frozen world initialization"
    if (
        energy is None
        and step == run.run.manifest.final_step
        and run.run.summary is not None
        and run.population_available
        and run.has_artifact(AGENTS_PATH)
        and run.report.components["summaries"] is CheckStatus.PASSED
    ):
        energy = _number(
            run.run.scan_agent_summaries()
            .filter(pl.col("alive_at_end"))
            .select(_finite_sum("final_energy"))
            .collect()
            .item()
        )
        energy_source = "terminal agent summaries"
    return (energy, energy_source), (resources, resource_source)


@dataclass(frozen=True, slots=True)
class EnergySummary:
    initial_energy_alive: Metric
    final_energy_alive: Metric
    feeding_energy_gained_total: Metric
    basal_energy_cost_total: Metric
    movement_energy_cost_total: Metric
    feeding_energy_cost_total: Metric
    reproduction_energy_cost_total: Metric
    gross_energy_cost_total: Metric
    offspring_energy_total: Metric
    net_reproductive_energy_cost_total: Metric
    energy_at_death_total: Metric
    positive_energy_removed_on_death: Metric
    expected_delta_energy: Metric
    observed_delta_energy: Metric
    energy_balance_residual: Metric
    births_per_resource_consumed: Metric
    feeding_energy_per_resource: Metric
    reproduction_cost_per_birth: Metric
    agent_steps_per_energy_cost: Metric


@dataclass(frozen=True, slots=True)
class ResourceSummary:
    initial_resources: Metric
    final_resources: Metric
    observed_min_resources: Metric
    observed_max_resources: Metric
    observed_mean_resources: Metric
    resource_consumed_total: Metric
    resource_regenerated_total: Metric


def _delta(final: float | None, initial: float | None) -> float | None:
    return None if final is None or initial is None else final - initial


def expected_energy_delta(
    feeding: float | None, gross: float | None, offspring: float | None, removed: float | None
) -> float | None:
    if feeding is None or gross is None or offspring is None or removed is None:
        return None
    return _number(feeding - gross + offspring - removed)


def energy_and_resources(
    run: AnalysisRun, window: AnalysisWindow | None, population: PopulationSummary | None
) -> tuple[EnergySummary, ResourceSummary]:
    coverage = run.bounds(window)
    start, end = coverage.start_step, coverage.end_step_exclusive
    final_capture = _checkpoint(run, end)
    initial_capture = final_capture if end == start else _checkpoint(run, start)
    initial_energy, initial_resources = _stocks(run, start, initial_capture)
    final_energy, final_resources = _stocks(run, end, final_capture)
    samples = (
        energy_series(run, window)
        .select(
            count().alias("samples"),
            *(_finite_sum(name).alias(name) for name in FLOWS),
            pl.col("total_resources").min().alias("minimum"),
            pl.col("total_resources").max().alias("maximum"),
            pl.col("total_resources").mean().alias("mean"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    complete = samples["samples"] == end - start
    accumulator = final_capture[2] if final_capture and start == run.coverage.start_step else {}

    def metric(
        value: float | None,
        source: str,
        limitation: str | None = None,
        *,
        sampled: bool = False,
    ) -> Metric:
        value = _number(value)
        quality = (
            Quality.UNAVAILABLE if value is None else Quality.SAMPLED if sampled else Quality.EXACT
        )
        return Metric(
            value,
            source,
            coverage,
            quality,
            limitation if value is None or sampled else None,
        )

    flows: dict[str, Metric] = {}
    for name in FLOWS:
        present = _number(samples[name]) if complete else None
        if present is not None:
            flows[name] = metric(present, "complete metrics")
        elif name in accumulator and _number(accumulator[name]) is not None and not complete:
            flows[name] = metric(accumulator[name], "checkpoint accumulator")
        else:
            flows[name] = metric(
                None, "metrics or checkpoint accumulator", "complete finite flux unavailable"
            )

    births, deaths = _event_energy(run, coverage)
    offspring = _number(births.select(_finite_sum("child_initial_energy")).collect().item())
    removed_value, positive_value = (
        deaths.select(
            _finite_sum("energy").alias("signed"), _positive_death_energy().alias("positive")
        )
        .collect()
        .row(0)
    )
    removed = _number(removed_value)
    costs = [flows[name].value for name in COSTS]
    gross = sum(cast(list[float], costs)) if None not in costs else None
    reproductive = _delta(flows["reproduction_energy_cost"].value, offspring)
    feeding = flows["feeding_energy_gained"].value
    consumed = flows["resource_consumed"].value
    observed = _delta(final_energy[0], initial_energy[0])
    expected = expected_energy_delta(feeding, gross, offspring, removed)
    saturated = bool(final_capture and final_capture[3])
    residual = None if saturated else _delta(observed, expected)
    resource_delta = _delta(final_resources[0], initial_resources[0])
    regenerated = None if consumed is None or resource_delta is None else resource_delta + consumed

    def derived_ratio(numerator: float | None, denominator: float | None, source: str) -> Metric:
        value = numerator / denominator if numerator is not None and denominator else None
        return metric(value, source, "missing numerator or missing or zero denominator")

    births_total = float(births.select(count()).collect().item())
    event_limitation = "nonfinite event energy prevents a complete total"
    flux_limitation = "one or more required terms unavailable"
    energy = EnergySummary(
        metric(*initial_energy, "initial state stock unavailable"),
        metric(*final_energy, "final state stock unavailable"),
        flows["feeding_energy_gained"],
        flows["basal_energy_cost"],
        flows["movement_energy_cost"],
        flows["feeding_energy_cost"],
        flows["reproduction_energy_cost"],
        metric(gross, "energy debits", flux_limitation),
        metric(offspring, "birth events", event_limitation),
        metric(reproductive, "reproduction debit and birth events", flux_limitation),
        metric(removed, "signed death events", event_limitation),
        metric(_number(positive_value), "positive energy in death events", event_limitation),
        metric(expected, "energy flows and events", flux_limitation),
        metric(observed, "endpoint energy stocks", flux_limitation),
        metric(
            residual,
            "endpoint energy stocks and energy budget",
            "saturated checkpoint accumulator" if saturated else flux_limitation,
        ),
        derived_ratio(births_total, consumed, "birth events and resource debit"),
        derived_ratio(feeding, consumed, "feeding credit and resource debit"),
        derived_ratio(
            flows["reproduction_energy_cost"].value,
            births_total,
            "reproduction debit and birth events",
        ),
        derived_ratio(
            None if population is None else float(population.agent_steps_at_start),
            gross,
            "population exposure and energy debits",
        ),
    )

    def resource_stat(name: str) -> Metric:
        return metric(
            _number(samples[name]),
            "metric observations",
            "statistics describe recorded samples only"
            if samples["samples"]
            else "no stock observations",
            sampled=not complete,
        )

    resources = ResourceSummary(
        metric(*initial_resources, "initial state stock unavailable"),
        metric(*final_resources, "final state stock unavailable"),
        resource_stat("minimum"),
        resource_stat("maximum"),
        resource_stat("mean"),
        flows["resource_consumed"],
        metric(regenerated, "resource stock change plus complete consumption", flux_limitation),
    )
    return energy, resources
