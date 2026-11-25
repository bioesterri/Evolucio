"""Reference values for the host analysis contracts."""

from types import SimpleNamespace
from typing import cast

import polars as pl
import pytest

from evolucio.analysis.energy import expected_energy_delta
from evolucio.analysis.mortality import (
    death_age_distribution,
    death_energy_distribution,
    death_energy_histogram,
    mortality_by_cause,
    mortality_by_phase,
    mortality_by_window,
)
from evolucio.analysis.population import bottlenecks, population_by_window, population_series
from evolucio.analysis.survival import kaplan_meier, median_survival_age
from evolucio.analysis.types import AnalysisPolicy, AnalysisWindow, Coverage
from evolucio.analysis.windows import AnalysisRun
from evolucio.core.codes import DeathCauseCode
from evolucio.persistence.loaders.run_data import RunData
from evolucio.persistence.validation.types import CheckStatus, ValidationReport


def trajectory(
    initial: int,
    births: list[int],
    deaths: list[int],
    end: int,
) -> AnalysisRun:
    source = SimpleNamespace(
        scan_founders=lambda: pl.LazyFrame({"agent_id": list(range(initial))}),
        scan_births=lambda: pl.LazyFrame({"birth_step": births}, schema={"birth_step": pl.Int64}),
        scan_deaths=lambda: pl.LazyFrame(
            {
                "death_step": deaths,
                "cause_code": [
                    int(DeathCauseCode.INVALID_STATE) if i % 2 else 1 for i in range(len(deaths))
                ],
                "phase_code": [1 + i % 2 for i in range(len(deaths))],
                "age": list(range(1, 1 + len(deaths))),
                "energy": [-2.0 if not i % 2 else float("nan") for i in range(len(deaths))],
            },
            schema=dict.fromkeys(("death_step", "cause_code", "phase_code", "age"), pl.Int64)
            | {"energy": pl.Float64},
        ),
        manifest=SimpleNamespace(
            run_id="test",
            artifacts=[SimpleNamespace(relative_path="artifacts/analysis/founders.parquet")],
        ),
    )
    report = SimpleNamespace(components={"population": CheckStatus.PASSED})
    return AnalysisRun(cast(RunData, source), cast(ValidationReport, report), Coverage(0, end), 10)


def test_demography_and_clipped_absolute_windows() -> None:
    run = trajectory(10, [0, 0, 0, 2], [0, 1, 1], 4)
    series = population_series(run).collect()
    assert series["population_after"].to_list() == [12, 10, 11, 11]
    assert series["population_before"].to_list() == [10, 12, 10, 11]
    assert series["birth_count"].sum() == 4
    assert series["death_count"].sum() == 3
    assert series["birth_fraction_per_step"].to_list() == [0.3, 0.0, 0.1, 0.0]
    windows = population_by_window(run, 3, AnalysisWindow(1, 4)).collect()
    assert windows["window_start"].to_list() == [0, 3]
    assert windows["covered_start"].to_list() == [1, 3]
    assert windows["window_end_exclusive"].to_list() == [3, 4]
    assert windows["steps_covered"].to_list() == [2, 1]
    assert windows["window_incomplete"].to_list() == [True, True]
    assert windows["agent_steps_at_start"].to_list() == [22, 11]
    assert windows["deaths_per_agent_step"].to_list() == [2 / 22, 0]


def test_bottlenecks_and_zero_denominators() -> None:
    run = trajectory(10, [1, 1, 1, 1], [0] * 8 + [2] * 5, 5)
    episodes = bottlenecks(run, AnalysisPolicy(bottleneck_fraction=0.5)).collect()
    assert episodes["episode_start_state_step"].to_list() == [1, 3]
    assert episodes["episode_end_state_step_exclusive"].to_list() == [2, 6]
    assert episodes["duration_observed"].to_list() == [1, 3]
    assert episodes["minimum_population"].to_list() == [2, 1]
    assert episodes["right_censored"].to_list() == [False, True]
    empty = trajectory(0, [], [], 2)
    assert population_series(empty).collect()["death_fraction_per_step"].null_count() == 2
    assert mortality_by_cause(empty).collect()["death_fraction"].null_count() == 5


def test_mortality_keeps_invalid_state_and_signed_energy() -> None:
    run = trajectory(10, [], [0, 1, 1], 3)
    causes = mortality_by_cause(run).collect()
    assert causes["death_count"].sum() == 3
    assert causes["death_fraction"].sum() == 1.0
    phases = mortality_by_phase(run).collect()
    assert phases["phase_name"].to_list() == ["PRE_ACTION", "POST_ACTION"]
    windows = mortality_by_window(run, 2).collect()
    assert windows.height == 10
    assert windows["death_count"].sum() == 3
    ages = death_age_distribution(run).collect().row(0, named=True)
    assert ages["count"] == 3 and ages["median"] == 2 and ages["q25"] == 1.5
    energy = death_energy_distribution(run).collect()
    assert energy["valid_energy_count"].to_list() == [2, 0]
    assert energy["invalid_energy_count"].to_list() == [0, 1]
    assert energy["mean"].to_list() == [-2.0, None]
    histogram = death_energy_histogram(run, [-2, 0]).collect()
    assert histogram["count"].to_list() == [0, 2, 0]
    assert histogram["invalid_count"].to_list() == [1, 1, 1]


def test_energy_reference() -> None:
    gross = 10 + 3 + 2 + 15
    assert 100 + expected_energy_delta(20, gross, 8, 4) == 94
    assert 100 + expected_energy_delta(20, gross, 8, -4) == 102
    assert expected_energy_delta(20, None, 8, 4) is None


def test_kaplan_meier_reference_and_ties() -> None:
    observations = pl.LazyFrame(
        {"observed_age": [2, 3, 4, 5], "event_observed": [True, False, True, False]}
    )
    curve = kaplan_meier(observations)
    frame = curve.collect()
    assert frame["n_at_risk"].to_list() == [4, 3, 2, 1]
    assert frame["survival_probability"].to_list() == [0.75, 0.75, 0.375, 0.375]
    assert median_survival_age(curve) == 4
    ties = kaplan_meier(observations.with_columns(pl.lit(2).alias("observed_age"))).collect()
    assert ties.row(0, named=True) == {
        "age": 2,
        "death_count": 2,
        "censored_count": 2,
        "n_at_risk": 4,
        "survival_probability": 0.5,
    }
    assert (
        median_survival_age(
            kaplan_meier(observations.limit(1).with_columns(pl.lit(False).alias("event_observed")))
        )
        is None
    )


def test_invalid_policy_and_window() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        AnalysisPolicy(window_steps=True)
    with pytest.raises(ValueError, match="between"):
        AnalysisPolicy(bottleneck_fraction=float("nan"))
    with pytest.raises(ValueError, match="start_step"):
        AnalysisWindow(2, 2)
    with pytest.raises(ValueError, match="outside committed"):
        population_series(trajectory(1, [], [], 3), AnalysisWindow(0, 4))
