"""PostgreSQL indexing and transaction rollback through the recorded run CLI."""

from pathlib import Path
from uuid import UUID

import polars as pl
import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from evolucio.analysis.summaries import AGENTS_PATH, LINEAGES_PATH
from evolucio.cli.run_command import run_config
from evolucio.persistence.genomes import GenomeHighlightReason, read_genome_index
from evolucio.persistence.local import RunStatus, load_run_manifest
from evolucio.persistence.postgres import (
    PostgresRegistry,
    create_session_factory,
    reconcile_run,
)
from evolucio.persistence.postgres import analysis as postgres_analysis
from evolucio.persistence.postgres.models import AgentSummaryModel, ArtifactModel
from evolucio.persistence.zarr.schemas import ZarrCapturePolicy
from evolucio.runtime import initialization
from tests.zarr_helpers import checkpoint_host_config, checkpoint_scenario


def test_registry_roundtrip_and_atomic_failure(
    tmp_path: Path,
    migrated_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _ = checkpoint_scenario()
    monkeypatch.setattr(initialization, "initialize_simulation", lambda *_a, **_kw: state)
    config_path = tmp_path / "config.json"
    config_path.write_text(checkpoint_host_config().model_dump_json())

    sessions = create_session_factory(migrated_engine)
    registry = PostgresRegistry(sessions)
    experiment = registry.create_experiment(name="PR-39 lifecycle projections")

    def record(output_root: Path) -> None:
        run_config(
            config_path,
            steps=3,
            chunk_steps=2,
            quiet=True,
            dry_run=False,
            output_root=output_root,
            experiment_id=experiment.experiment_id,
            capture_policy=ZarrCapturePolicy(checkpoint_on_completion=True),
        )

    record(tmp_path / "completed")
    run = registry.list_runs(experiment_id=experiment.experiment_id)[0]
    assert run.status == "completed"
    run_dir = tmp_path / "completed" / str(run.run_id)
    assert reconcile_run(run_dir, registry).ok
    comparison = registry.list_runs_for_comparison(experiment.experiment_id)
    assert comparison[0].run_id == run.run_id and comparison[0].summary is not None
    assert len(registry.list_state_captures(run.run_id)) == 1
    agents = pl.read_parquet(run_dir / AGENTS_PATH)
    lineages = pl.read_parquet(run_dir / LINEAGES_PATH)
    for row in agents.iter_rows(named=True):
        stored = registry.get_agent_summary(run.run_id, row["agent_id"])
        assert stored is not None
        assert all(
            getattr(stored, name) == value for name, value in row.items() if name != "run_id"
        )
    for row in lineages.iter_rows(named=True):
        lineage_id = row["lineage_id"]
        stored = registry.get_lineage_summary(run.run_id, lineage_id)
        assert stored is not None
        assert all(
            getattr(stored, name) == value for name, value in row.items() if name != "run_id"
        )
        assert len(registry.list_agents_by_lineage(run.run_id, lineage_id)) == row["total_agents"]
    highlights = registry.get_highlighted_genomes(run.run_id)
    index = read_genome_index(run_dir)
    assert len(highlights) == len(index.genomes)
    assert sum(len(entry.highlights) for entry in highlights) == sum(
        len(genome.reasons) for genome in index.genomes
    )
    assert len(registry.get_genomes_by_reason(run.run_id, GenomeHighlightReason.FOUNDER)) == 5
    assert len(registry.list_lineages(run.run_id, order_by="total_descendants")) == lineages.height
    assert registry.lineage_outcomes_for_experiment(experiment.experiment_id)[0][3:] == (
        lineages.height,
        lineages["max_generation"].max(),
        lineages["total_agents"].max(),
    )

    original = postgres_analysis.insert_run_analysis

    def fail(session: Session, run_id: UUID, run_dir: Path) -> None:
        original(session, run_id, run_dir)
        raise RuntimeError("forced analysis transaction failure")

    monkeypatch.setattr(postgres_analysis, "insert_run_analysis", fail)
    with pytest.raises(RuntimeError, match="forced analysis transaction failure"):
        record(tmp_path / "failed")
    failed_dir = next((tmp_path / "failed").iterdir())
    failed = load_run_manifest(failed_dir / "manifest.json")
    failed_run_id = UUID(failed.run_id)
    stored_failure = registry.get_run(failed_run_id)
    assert stored_failure is not None and stored_failure.status == "failed"
    assert failed.status is RunStatus.FAILED
    assert not (failed_dir / AGENTS_PATH).exists()
    assert not (failed_dir / LINEAGES_PATH).exists()
    assert not (failed_dir / "summary.json").exists()
    assert registry.get_highlighted_genomes(failed_run_id) == ()
    assert reconcile_run(failed_dir, registry).ok
    with sessions() as session:
        assert session.scalar(select(AgentSummaryModel).filter_by(run_id=failed_run_id)) is None
        paths = set(
            session.scalars(select(ArtifactModel.relative_path).filter_by(run_id=failed_run_id))
        )
        assert "error.json" in paths
        assert not {AGENTS_PATH, LINEAGES_PATH, "summary.json"} & paths
