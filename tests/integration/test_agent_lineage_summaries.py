"""Recorded lifecycle projections with births, deaths and surviving lineages."""

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from evolucio.analysis.summaries import AGENTS_PATH, LINEAGES_PATH
from evolucio.persistence.genomes import GenomeHighlightReason, read_genome_index
from evolucio.persistence.loaders.run_data import open_run
from evolucio.persistence.local import verify_local_run
from evolucio.persistence.validation.runner import validate_run
from evolucio.persistence.validation.types import ValidationLevel
from evolucio.runtime import RunMode, RunPlan, initialization
from evolucio.runtime.recorded_run import run_recorded_simulation
from tests.zarr_helpers import checkpoint_host_config, checkpoint_scenario


def test_completed_lifecycle_and_chunk_neutrality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, compiled = checkpoint_scenario()
    monkeypatch.setattr(initialization, "initialize_simulation", lambda *_a, **_kw: state)
    frames: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    selections = []
    for chunk, mode in (
        (2, RunMode.EXPERIMENT),
        (3, RunMode.EXPERIMENT),
        (2, RunMode.BENCHMARK),
    ):
        run = run_recorded_simulation(
            config=checkpoint_host_config(),
            compiled=compiled,
            output_root=tmp_path,
            plan=RunPlan(3, chunk),
            mode=mode,
        )
        assert verify_local_run(run.run_dir).ok
        agents = pl.read_parquet(run.run_dir / AGENTS_PATH)
        lineages = pl.read_parquet(run.run_dir / LINEAGES_PATH)
        assert run.summary.births_total > 0 and run.summary.deaths_total > 0
        assert agents.height == 5 + run.summary.births_total
        assert agents.filter(~pl.col("alive_at_end")).height == run.summary.deaths_total
        assert lineages["total_agents"].sum() == agents.height
        assert lineages["alive_final"].sum() == run.summary.alive_count
        assert lineages["births_total"].sum() == run.summary.births_total

        index = read_genome_index(run.run_dir)
        representatives = [
            genome.lineage_id
            for genome in index.genomes
            if GenomeHighlightReason.FINAL_LINEAGE_REPRESENTATIVE in genome.reasons
        ]
        assert (
            sorted(representatives)
            == lineages.filter(pl.col("alive_final") > 0)["lineage_id"].to_list()
        )
        assert lineages.filter(pl.col("alive_final") == 0).height > 0
        assert sum(GenomeHighlightReason.FOUNDER in g.reasons for g in index.genomes) == 5
        frames.append((agents.drop("run_id"), lineages.drop("run_id")))
        selections.append(index.genomes)
    expected = [(g.genome_id, g.genome_hash, g.reasons) for g in selections[0]]
    for other_frames, other_selection in zip(frames[1:], selections[1:], strict=True):
        for first, second in zip(frames[0], other_frames, strict=True):
            assert_frame_equal(first, second, check_exact=True)
        assert [(g.genome_id, g.genome_hash, g.reasons) for g in other_selection] == expected


def test_total_extinction_keeps_founders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import equinox as eqx
    import jax
    import numpy as np

    state, _ = checkpoint_scenario()
    energy = np.asarray(state.population.energy).copy()
    energy[np.asarray(state.population.alive)] = 0.01
    state = eqx.tree_at(lambda value: value.population.energy, state, jax.device_put(energy))
    monkeypatch.setattr(initialization, "initialize_simulation", lambda *_a, **_kw: state)
    config = checkpoint_host_config()
    config = config.model_copy(
        update={"population": config.population.model_copy(update={"initial_agents": 5})}
    )
    run = run_recorded_simulation(
        config=config,
        output_root=tmp_path,
        plan=RunPlan(1, 1),
    )
    assert run.summary.alive_count == 0
    lineages = pl.read_parquet(run.run_dir / LINEAGES_PATH)
    assert lineages.height == 5
    assert lineages["extinction_step"].to_list() == [0] * 5
    assert lineages["final_population_fraction"].to_list() == [0.0] * 5
    index = read_genome_index(run.run_dir)
    assert len(index.genomes) == 5
    assert all(record.reasons == {GenomeHighlightReason.FOUNDER: 0} for record in index.genomes)
    report = validate_run(run.run_dir, validation_level=ValidationLevel.SCIENTIFIC)
    assert report.validation_passed, report.issues
    assert open_run(run.run_dir).scan_births().collect().height == 0
