"""Analyze real persisted trajectories at multiple capture granularities."""

import json
from pathlib import Path

import equinox as eqx
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from evolucio.analysis.energy import energy_by_window, energy_series
from evolucio.analysis.export import export_analysis
from evolucio.analysis.genealogy import genealogy_edges, genealogy_nodes
from evolucio.analysis.genetic_diversity import (
    genetic_diversity_at_checkpoint,
    genetic_diversity_series,
)
from evolucio.analysis.genome_distance import (
    founder_genome_divergence,
    parent_child_genome_divergence,
    retained_genome,
)
from evolucio.analysis.lineage_analysis import analyze_genetic_diversity, analyze_lineages
from evolucio.analysis.lineage_diversity import lineage_diversity_series
from evolucio.analysis.lineage_dynamics import (
    lineage_extinction_summary,
    lineage_population_by_window,
    lineage_population_series,
)
from evolucio.analysis.mutation_analysis import mutation_summary_by_lineage
from evolucio.analysis.offspring import offspring_distribution, reproductive_success_by_agent
from evolucio.analysis.population import population_series
from evolucio.analysis.summary import analyze_run
from evolucio.analysis.survival import cohort_summary, survival_observations
from evolucio.analysis.types import (
    AnalysisPolicy,
    AnalysisUnavailableError,
    AnalysisWindow,
    Quality,
)
from evolucio.analysis.windows import validated
from evolucio.cli.main import app
from evolucio.persistence.loaders.run_data import open_run
from evolucio.persistence.local.artifacts import sha256_file
from evolucio.persistence.parquet.schemas import ParquetPersistencePolicy
from evolucio.persistence.zarr.schemas import ZarrCapturePolicy
from evolucio.runtime.recorded_run import run_recorded_simulation
from evolucio.runtime.types import RunPlan
from tests.regression.helpers import FLOAT32_ATOL, FLOAT32_RTOL
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
    lineage_frames: list[list[pl.DataFrame]] = []
    recorded = None
    for chunk in (1, 3):
        recorded = run_recorded_simulation(
            config=config,
            output_root=tmp_path,
            plan=RunPlan(3, chunk),
            capture_policy=ZarrCapturePolicy(checkpoint_on_completion=True),
        )
        data = validated(open_run(recorded.run_dir))
        results.append(analyze_run(data))
        lineage_frames.append(
            [
                frame.collect().drop("run_id", strict=False)
                for frame in (
                    genealogy_nodes(data),
                    genealogy_edges(data),
                    offspring_distribution(data).histogram,
                    lineage_population_series(data),
                    lineage_diversity_series(data),
                    mutation_summary_by_lineage(data),
                    genetic_diversity_series(data),
                )
            ]
        )
    for first_frame, second_frame in zip(*lineage_frames, strict=True):
        assert_frame_equal(first_frame, second_frame, rel_tol=FLOAT32_RTOL, abs_tol=FLOAT32_ATOL)

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
    data = validated(loaded)
    lineages = analyze_lineages(data)
    assert set(lineages.summaries) == {"quality", "diversity", "mutations", "offspring", "lineages"}
    checkpoint = genetic_diversity_at_checkpoint(data, 3, distinct_contents=True)
    assert checkpoint.population == result.population_summary.final_population
    assert checkpoint.total_parameter_variance == pytest.approx(
        checkpoint.within_lineage_variance + checkpoint.between_lineage_variance
    )
    assert checkpoint.lineages["alive_count"].sum() == checkpoint.population
    dense = lineage_population_series(data, dense=True).collect()
    assert (
        dense.group_by("step")
        .agg(pl.col("population_after").sum())
        .sort("step")["population_after"]
        .to_list()
        == population_series(data).collect()["population_after"].to_list()
    )
    windows = lineage_population_by_window(data, 2).collect()
    assert windows["births_total"].sum() == births
    assert windows["deaths_total"].sum() == deaths
    assert windows["agent_steps_at_start"].sum() == dense["population_before"].sum()
    assert lineage_extinction_summary(data).filter(pl.col("extinction_observed")).collect().height
    assert reproductive_success_by_agent(data).collect()["direct_offspring_count"].sum() == births
    assert founder_genome_divergence(data).collect().height
    highlighted_ids = [record.genome_id for record in loaded.list_highlighted_genomes()]
    captured_only = (
        reproductive_success_by_agent(data)
        .filter(pl.col("alive_at_end") & ~pl.col("genome_id").is_in(highlighted_ids))
        .select("genome_id")
        .collect()["genome_id"]
        .to_list()
    )
    assert captured_only
    with pytest.raises(AnalysisUnavailableError):
        retained_genome(data, captured_only[0])
    assert retained_genome(data, captured_only[0], state_step=3)
    for child_id in data.run.scan_births().collect()["child_agent_id"]:
        comparison = parent_child_genome_divergence(data, child_id)
        if comparison["quality"] is Quality.EXACT:
            assert comparison["mutation_deltas_match"]
    lineage_output = export_analysis(data, tmp_path / "derived-lineages", include_lineages=True)
    lineage_payload = json.loads((lineage_output / "lineage_analysis_summary.json").read_text())
    assert set(lineage_payload["exported_tables"]) == {
        "lineage_population_series",
        "lineage_diversity_series",
        "offspring_distribution",
        "generation_summary",
        "lineage_extinction_summary",
        "genetic_diversity_series",
        "mutation_summary_by_lineage",
        "founder_genome_divergence",
    }
    for command, options in (
        ("lineages", []),
        ("offspring", []),
        ("generations", []),
        ("diversity", ["--state-step", "3"]),
        ("mutations", []),
        ("ancestors", ["--agent-id", "0"]),
        ("descendants", ["--agent-id", "0", "--max-depth", "1"]),
        ("genome-distance", ["--genome-a", "0", "--genome-b", "1"]),
    ):
        command_result = CliRunner().invoke(app, ["analysis", command, str(loaded.path), *options])
        assert command_result.exit_code == 0, (command, command_result.exception)
    runner = CliRunner()
    for arguments in (
        ["summary"],
        ["population"],
        ["energy"],
        ["survival"],
        ["mortality"],
        ["lineages", str(loaded.path), "--start-step", "1", "--end-step-exclusive", "2"],
        ["diversity", str(loaded.path), "--start-step", "1", "--end-step-exclusive", "2"],
        ["genome-distance", str(loaded.path), "--genome-a", "0", "--genome-b", "999"],
        [
            "export",
            str(loaded.path),
            "--lineages",
            "--output-dir",
            str(tmp_path / "window"),
            "--start-step",
            "1",
            "--end-step-exclusive",
            "2",
        ],
    ):
        if len(arguments) == 1:
            arguments.append(str(loaded.path))
        result_cli = runner.invoke(app, ["analysis", *arguments])
        assert result_cli.exit_code == 0, (arguments, result_cli.exception)
        if arguments[0] == "genome-distance":
            assert json.loads(result_cli.output)["quality"] == "UNAVAILABLE"
    invalid_window = runner.invoke(
        app,
        ["analysis", "diversity", str(loaded.path), "--start-step", "1"],
    )
    assert invalid_window.exit_code == 2
    incompatible_window = runner.invoke(
        app,
        [
            "analysis",
            "diversity",
            str(loaded.path),
            "--start-step",
            "1",
            "--end-step-exclusive",
            "2",
            "--state-step",
            "3",
        ],
    )
    assert incompatible_window.exit_code == 2
    window_payload = json.loads((tmp_path / "window" / "lineage_analysis_summary.json").read_text())
    assert "lifecycle_summaries" in window_payload["unavailable_metrics"]
    assert "offspring_distribution" not in window_payload["exported_tables"]
    with pytest.raises(AnalysisUnavailableError, match="outside the verified prefix"):
        genetic_diversity_at_checkpoint(data, 4)
    with pytest.raises(AnalysisUnavailableError, match="validated checkpoint"):
        genetic_diversity_at_checkpoint(data, 1)
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
        data = validated(loaded)
        diversity = lineage_diversity_series(data).collect()
        assert diversity["step"].to_list() == [0, 1]
        assert diversity["active_lineages"].to_list() == [4, 0]
        assert diversity["shannon_entropy"][1] is None
        assert genetic_diversity_series(data).collect()["step"].to_list() == [0, 10]
        assert analyze_genetic_diversity(data).summaries["temporal_quality"] is Quality.SAMPLED
        unobserved = analyze_genetic_diversity(data, window=AnalysisWindow(2, 9))
        assert unobserved.summaries["temporal_quality"] is Quality.UNAVAILABLE
        assert "genetic_parameter_variance" in unobserved.unavailable_metrics
        assert lineage_diversity_series(data, AnalysisWindow(2, 9)).collect()["step"].to_list() == [
            2
        ]
        if checkpoint:
            assert genetic_diversity_at_checkpoint(data, 12).population == 0
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


def test_recorded_multigeneration_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolucio.analysis.offspring import generation_summary
    from evolucio.core.codes import ActionCode
    from evolucio.core.observations.schema import ObservationIndex
    from evolucio.persistence.validation.runner import validate_run
    from evolucio.persistence.validation.types import ValidationLevel

    state, _ = checkpoint_scenario()
    # The founder and descendants eat below the reproduction threshold and reproduce above it.
    genomes = state.genomes
    weights1 = genomes.layer1.weight.at[0, 0, ObservationIndex.REPRODUCTION_MARGIN].set(4.0)
    weights2 = genomes.layer2.weight.at[0, int(ActionCode.REPRODUCE), 0].set(4.0)
    weights2 = weights2.at[0, int(ActionCode.EAT), 0].set(-4.0)
    biases2 = genomes.layer2.bias.at[0, int(ActionCode.REPRODUCE)].set(0.0)
    genomes = eqx.tree_at(
        lambda genome: (genome.layer1.weight, genome.layer2.weight, genome.layer2.bias),
        genomes,
        (weights1, weights2, biases2),
    )
    state = eqx.tree_at(lambda value: value.genomes, state, genomes)
    monkeypatch.setattr(
        "evolucio.runtime.initialization.initialize_simulation", lambda *_a, **_kw: state
    )
    config = checkpoint_host_config()
    config = config.model_copy(
        update={
            "population": config.population.model_copy(update={"initial_agents": 5}),
            "energy": config.energy.model_copy(
                update={
                    "reproduction_threshold": 6.0,
                    "reproduction_cost": 5.0,
                    "offspring_initial_energy": 5.0,
                }
            ),
            "evolution": config.evolution.model_copy(
                update={"weight_mutation_sigma": 0.005, "bias_mutation_sigma": 0.005}
            ),
        }
    )
    recorded = run_recorded_simulation(
        config=config,
        output_root=tmp_path,
        plan=RunPlan(6, 3),
        capture_policy=ZarrCapturePolicy(checkpoint_on_completion=True),
    )
    report = validate_run(recorded.run_dir, validation_level=ValidationLevel.SCIENTIFIC)
    assert report.validation_passed, report.issues
    data = validated(open_run(recorded.run_dir))
    generations = generation_summary(data).collect()
    assert generations["generation"].max() >= 2
    assert generations["mutation_effective_total"].sum() > 0
    assert analyze_lineages(data).summaries["lineages"] == {
        "lineages_total": 5,
        "lineages_extinct": 1,
        "lineages_alive_at_prefix_end": 4,
    }
    population = analyze_run(data).population_summary
    assert population is not None
    checkpoint = genetic_diversity_at_checkpoint(data, 6)
    assert checkpoint.population == population.final_population
    assert founder_genome_divergence(data).collect().height == 4
