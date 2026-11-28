"""Tests for the frozen design, seed pairing and descriptive statistics."""

from pathlib import Path

import polars as pl
import pytest

from evolucio.analysis.comparison.statistics import (
    BootstrapPolicy,
    bootstrap_paired_mean,
    compare_paired_metrics,
    extinction_tables,
    summarize_case_metrics,
    summarize_paired,
)
from evolucio.config import load_config
from evolucio.experiments.config_diff import (
    DifferenceCategory,
    compare_configs,
    configuration_group_hash,
)
from evolucio.experiments.design import load_design, resolve_design

EXAMPLE = Path("examples/experimental_comparison/design.json")


def test_design_matrix_and_controls() -> None:
    design = load_design(EXAMPLE)
    resolved = resolve_design(design)
    assert resolved.batch.preflight.run_count == 12
    assert all(not issues for issues in resolved.blocking_errors.values())
    three_cases = design.model_copy(
        update={
            "cases": design.cases[:3],
            "allowed_config_differences": {
                key: value
                for key, value in design.allowed_config_differences.items()
                if key != "abundant_resources"
            },
        }
    )
    assert resolve_design(three_cases).batch.preflight.run_count == 9
    for case in design.cases:
        assert {spec.seed for spec in resolved.batch.specs if spec.case_id == case.case_id} == {
            11,
            22,
            33,
        }
    assert {d.parameter_path for d in resolved.differences["no_mutation"]} == {
        "evolution.weight_mutation_rate",
        "evolution.bias_mutation_rate",
    }
    (constant,) = resolved.differences["constant_environment"]
    assert constant.parameter_path == "world.environment_schedule"
    assert constant.variant_value == []
    invalid = design.model_copy(
        update={
            "allowed_config_differences": design.allowed_config_differences | {"no_mutation": ()}
        }
    )
    assert len(resolve_design(invalid).blocking_errors["no_mutation"]) == 2


def test_unplanned_metabolism_change_is_reported(tmp_path: Path) -> None:
    design = load_design(EXAMPLE)
    case = design.cases[1]
    config = load_config(case.config_path)
    changed = config.model_copy(
        update={
            "energy": config.energy.model_copy(update={"basal_cost": config.energy.basal_cost * 2}),
        }
    )
    path = tmp_path / "changed.json"
    path.write_text(changed.model_dump_json(), encoding="utf-8")
    altered = design.model_copy(
        update={
            "cases": (
                design.cases[0],
                case.model_copy(update={"config_path": path}),
                *design.cases[2:],
            ),
        }
    )
    resolved = resolve_design(altered)
    assert resolved.blocking_errors[case.case_id] == ("Uncontrolled parameter: energy.basal_cost",)
    assert {diff.parameter_path for diff in resolved.differences[case.case_id]} == {
        "energy.basal_cost",
        "evolution.weight_mutation_rate",
        "evolution.bias_mutation_rate",
    }


def test_projection_retains_scientific_and_capture_fields() -> None:
    config = load_config(EXAMPLE.with_name("baseline.json"))
    other = config.model_copy(
        update={
            "seed": 11,
            "runtime": config.runtime.model_copy(update={"chunk_size": 2}),
        }
    )
    assert configuration_group_hash(config) == configuration_group_hash(other)
    assert {change.difference_category for change in compare_configs(config, other)} == {
        DifferenceCategory.OPERATIONAL,
        DifferenceCategory.SEED,
    }
    for update in ({"record_stride": 2}, {"backend": "auto"}):
        changed = config.model_copy(update={"runtime": config.runtime.model_copy(update=update)})
        assert configuration_group_hash(config) != configuration_group_hash(changed)


def _observations(seeds: list[int], values: list[float | None]) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "seed": seeds,
            "run_id": [str(seed) for seed in seeds],
            "metric_name": "final_population",
            "dimension": "",
            "metric_value": values,
            "availability_status": "defined",
            "case_id": "sample",
        }
    )


def test_paired_statistics_and_missing_seed() -> None:
    baseline = _observations([11, 22, 33], [10, 20, 30])
    variant = _observations([33, 11, 22], [39, 15, 18])
    paired = compare_paired_metrics(baseline, variant, "final_population")
    assert paired.collect()["absolute_difference"].to_list() == [5, -2, 9]
    summary = summarize_paired(paired.with_columns(pl.lit("variant").alias("variant_case_id")), 0)
    row = summary.collect().row(0, named=True)
    assert (row["n_pairs"], row["mean_difference"], row["median_difference"]) == (3, 4, 5)
    assert (row["positive_difference_count"], row["negative_difference_count"]) == (2, 1)
    missing = compare_paired_metrics(
        baseline, variant.filter(pl.col("seed") != 22), "final_population"
    )
    assert missing.collect()["absolute_difference"].to_list() == [5, None, 9]
    with pytest.raises(pl.exceptions.ComputeError, match="join keys"):
        compare_paired_metrics(
            baseline, pl.concat([variant, variant]), "final_population"
        ).collect()
    row = summarize_case_metrics(_observations([11, 22], [0, None])).collect().row(0, named=True)
    assert (row["n_expected"], row["n_missing"], row["sample_std"]) == (2, 1, None)


def test_bootstrap_and_extinction_censoring() -> None:
    policy = BootstrapPolicy(9, 150)
    assert bootstrap_paired_mean([5, -2, 9], policy) == bootstrap_paired_mean([5, -2, 9], policy)
    assert bootstrap_paired_mean([2, 2, 2], policy) == (2, 2)
    assert bootstrap_paired_mean([2], policy) is None
    with pytest.raises(ValueError):
        BootstrapPolicy(9, 0)
    metrics = pl.LazyFrame(
        {
            "metric_name": ["population_extinct"] * 3 + ["time_to_extinction"] * 3,
            "case_id": ["a"] * 6,
            "seed": [11, 22, 33] * 2,
            "run_id": ["one", "two", None] * 2,
            "metric_value": [1, 0, None, 2, None, None],
        }
    )
    tables = extinction_tables(metrics, pl.LazyFrame(), 5)
    summary = tables["extinction_summary"].collect().row(0, named=True)
    assert (
        summary["n_completed"],
        summary["n_extinct"],
        summary["observed_extinction_fraction"],
    ) == (2, 1, 0.5)
    assert tables["extinction_observations"].collect().select(
        "event_observed", "observed_age"
    ).rows() == [(True, 2), (False, 5)]
