"""Analyze real persisted trajectories at multiple capture granularities."""

import json
from pathlib import Path

import equinox as eqx
import polars as pl
import pytest
from typer.testing import CliRunner

from evolucio.analysis.energy import energy_by_window, energy_series
from evolucio.analysis.export import export_analysis
from evolucio.analysis.population import population_series
from evolucio.analysis.summary import analyze_run
from evolucio.analysis.survival import cohort_summary, survival_observations
from evolucio.analysis.types import AnalysisPolicy, AnalysisWindow, Quality
from evolucio.cli.main import app
from evolucio.persistence.loaders.run_data import open_run
from evolucio.persistence.local.artifacts import sha256_file
from evolucio.persistence.parquet.schemas import ParquetPersistencePolicy
from evolucio.persistence.zarr.schemas import ZarrCapturePolicy
from evolucio.runtime.recorded_run import run_recorded_simulation
from evolucio.runtime.types import RunPlan
from tests.zarr_helpers import checkpoint_host_config, checkpoint_scenario


def test_real_analysis_export_and_chunk_neutrality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _ = checkpoint_scenario()
    monkeypatch.setattr(
        "evolucio.runtime.initialization.initialize_simulation", lambda *_a, **_kw: state
    )
    config = checkpoint_host_config()
    config = config.model_copy(
        update={
            "population": config.population.model_copy(update={"initial_agents": 5}),
            "energy": config.energy.model_copy(
                update={"reproduction_cost": 15.0, "offspring_initial_energy": 8.0}
            ),
        }
    )
    results = []
    recorded = None
    for chunk in (1, 3):
        recorded = run_recorded_simulation(
            config=config,
            output_root=tmp_path,
            plan=RunPlan(3, chunk),
            capture_policy=ZarrCapturePolicy(checkpoint_on_completion=True),
        )
        results.append(analyze_run(open_run(recorded.run_dir)))
    assert recorded is not None
    first, result = results
    for name in (
        "population_summary",
        "energy_summary",
        "resource_summary",
        "mortality_summary",
        "survival_summary",
    ):
        assert getattr(first, name) == getattr(result, name), name
    loaded = open_run(recorded.run_dir)
    assert result.population_summary is not None and result.survival_summary is not None
    births = loaded.scan_births().collect().height
    deaths = loaded.scan_deaths().collect().height
    assert births > 0 and deaths > 0
    assert result.population_summary.births_total == births
    assert result.population_summary.final_population == 5 + births - deaths
    assert result.mortality_summary.deaths_total == deaths
    assert result.survival_summary.right_censored_count == 5 + births - deaths
    energy = result.energy_summary
    assert energy.energy_balance_residual.value == pytest.approx(0, abs=3e-5)
    assert energy.energy_at_death_total.value == pytest.approx(-0.05)
    assert energy.net_reproductive_energy_cost_total.value == 7.0 * births
    assert result.resource_summary.resource_consumed_total.value == pytest.approx(
        float(recorded.result.runtime.accumulator.resource_consumed_total)
    )
    observations = survival_observations(loaded).collect()
    assert observations.filter(~pl.col("event_observed"))["death_cause_code"].null_count() == (
        result.survival_summary.right_censored_count
    )
    assert cohort_summary(loaded).collect()["is_founder"].to_list() == [False, True]
    before = {path: sha256_file(path) for path in loaded.path.rglob("*") if path.is_file()}
    policy = AnalysisPolicy(window_steps=2)
    output = export_analysis(loaded, tmp_path / "derived", policy=policy)
    files = {path.name for path in output.iterdir()}
    assert len(files) == 8
    payload = json.loads((output / "analysis_summary.json").read_text())
    assert payload["analysis_policy"]["window_steps"] == 2
    assert payload["source_config_hash"] == loaded.manifest.config_hash
    assert payload["source_dataset_hashes"]
    assert pl.read_parquet(output / "population_by_window.parquet")["steps_covered"].to_list() == [
        2,
        1,
    ]
    assert before == {path: sha256_file(path) for path in loaded.path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="outside"):
        export_analysis(loaded, loaded.path / "derived")
    cli = CliRunner().invoke(app, ["analysis", "summary", str(loaded.path), "--format", "json"])
    assert cli.exit_code == 0, cli.exception
    assert json.loads(cli.output)["run_id"] == loaded.manifest.run_id


def test_sampled_extinction_and_accumulator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _ = checkpoint_scenario()
    # Suppress reproduction and feeding so max_age extinguishes the population at step 1.
    bias = state.genomes.layer2.bias * 0
    state = eqx.tree_at(lambda value: value.genomes.layer2.bias, state, bias.at[:5, 0].set(2))
    monkeypatch.setattr(
        "evolucio.runtime.initialization.initialize_simulation", lambda *_a, **_kw: state
    )
    config = checkpoint_host_config()
    config = config.model_copy(
        update={
            "population": config.population.model_copy(update={"initial_agents": 5}),
            "evolution": config.evolution.model_copy(update={"max_age": 2}),
        }
    )
    for checkpoint in (False, True):
        recorded = run_recorded_simulation(
            config=config,
            output_root=tmp_path,
            plan=RunPlan(12, 4),
            parquet_policy=ParquetPersistencePolicy(metrics_interval=10),
            capture_policy=ZarrCapturePolicy(checkpoint_on_completion=checkpoint),
        )
        loaded = open_run(recorded.run_dir)
        result = analyze_run(loaded)
        population = result.population_summary
        assert population is not None
        assert population.extinction_state_step == 2
        assert population.deaths_total == 5 and population.births_total == 0
        assert population_series(loaded).collect()["population_after"].to_list() == [4] + [0] * 11
        assert energy_series(loaded).collect()["step"].to_list() == [0, 10]
        assert result.resource_summary.observed_min_resources.quality is Quality.SAMPLED
        basal = result.energy_summary.basal_energy_cost_total
        window = energy_by_window(loaded, 100).collect()
        if checkpoint:
            assert basal.quality is Quality.EXACT and basal.source == "checkpoint accumulator"
            assert window["basal_energy_cost_total"].item() == basal.value
        else:
            assert basal.quality is Quality.UNAVAILABLE
            assert result.energy_summary.energy_balance_residual.value is None
            assert window["basal_energy_cost_total"].item() is None
        assert result.energy_summary.births_per_resource_consumed.value is None
        assert result.energy_summary.reproduction_cost_per_birth.value is None
        final = analyze_run(loaded, window=AnalysisWindow(2, 9))
        assert final.survival_summary is None
        assert (
            final.population_summary is not None and final.population_summary.population_cv is None
        )
        assert final.population_summary.extinction_state_step == 2
