# ruff: noqa: ANN001, ANN201, ANN202
import json
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from evolucio.config import load_config
from evolucio.persistence.local import create_local_run
from evolucio.persistence.postgres import (
    PostgresRegistry,
    RegistryIntegrityError,
    create_session_factory,
)
from evolucio.persistence.postgres.models import RunModel
from evolucio.runtime import RunMode, RunPlan


def _local_recorder(tmp_path):
    return create_local_run(
        config=load_config("tests/fixtures/config/valid_v1.json"),
        plan=RunPlan(4, 2),
        mode=RunMode.EXPERIMENT,
        output_root=tmp_path,
    )


def test_migration_schema_and_roundtrip(migrated_engine):
    inspector = inspect(migrated_engine)
    assert {
        "experiment",
        "run",
        "config_snapshot",
        "run_summary",
        "artifact",
        "dataset_index",
        "state_capture",
    } <= set(inspector.get_table_names())
    assert inspector.get_pk_constraint("run")["constrained_columns"] == ["run_id"]
    columns = {col["name"]: col["type"] for col in inspector.get_columns("run")}
    assert isinstance(columns["run_id"], PGUUID)
    assert isinstance(columns["created_at"], TIMESTAMP) and columns["created_at"].timezone
    config_columns = {col["name"]: col["type"] for col in inspector.get_columns("config_snapshot")}
    assert isinstance(config_columns["config_json"], JSONB)
    for table in ("config_snapshot", "run_summary", "artifact", "dataset_index", "state_capture"):
        (foreign_key,) = inspector.get_foreign_keys(table)
        assert foreign_key["options"]["ondelete"] == "CASCADE"
        assert foreign_key["referred_table"] == "run"
    assert {tuple(index["column_names"]) for index in inspector.get_indexes("run")} >= {
        ("experiment_id",),
        ("config_hash",),
        ("seed",),
        ("status",),
        ("created_at",),
        ("experiment_id", "config_hash"),
        ("config_hash", "seed"),
    }
    for table, columns in (
        ("artifact", ["run_id", "relative_path"]),
        ("dataset_index", ["run_id", "dataset_kind"]),
        ("state_capture", ["run_id", "capture_kind", "state_step"]),
    ):
        assert columns in [
            constraint["column_names"] for constraint in inspector.get_unique_constraints(table)
        ]
    registry = PostgresRegistry(create_session_factory(migrated_engine))
    created = registry.create_experiment(name="paired seeds", objective="compare cases")
    assert registry.get_experiment(created.experiment_id) == created
    assert registry.list_experiments() == (created,)


def test_experiment_delete_is_restricted(migrated_engine):
    # The FK is verified against PostgreSQL's catalog, not only Python metadata.
    fks = inspect(migrated_engine).get_foreign_keys("run")
    assert fks[0]["options"]["ondelete"] == "RESTRICT"


def test_database_constraints_are_real(migrated_engine):
    registry = PostgresRegistry(create_session_factory(migrated_engine))
    with pytest.raises(ValueError):
        registry.create_experiment(name=" ")


def test_pr35_identity_and_atomic_run_config_creation(migrated_engine, tmp_path):
    recorder = _local_recorder(tmp_path)
    config_path = recorder.run_dir / "config.json"
    config_json = json.loads(config_path.read_text())
    artifact = next(
        item for item in recorder.manifest.artifacts if item.relative_path == "config.json"
    )
    registry = PostgresRegistry(create_session_factory(migrated_engine))
    run = registry.create_run(
        manifest=recorder.manifest,
        config_json=config_json,
        config_schema_version=1,
        config_artifact_sha256=artifact.sha256,
        local_run_path=str(recorder.run_dir),
    )
    assert str(run.run_id) == recorder.run_id

    second = _local_recorder(tmp_path)
    second_path = second.run_dir / "config.json"
    second_json = json.loads(second_path.read_text())
    second_artifact = next(
        item for item in second.manifest.artifacts if item.relative_path == "config.json"
    )
    with pytest.raises(RegistryIntegrityError):
        registry.create_run(
            manifest=second.manifest,
            config_json=second_json,
            config_schema_version=None,  # type: ignore[arg-type]
            config_artifact_sha256=second_artifact.sha256,
        )
    with create_session_factory(migrated_engine)() as session:
        assert session.get(RunModel, UUID(second.manifest.run_id)) is None


def test_completed_recorded_run_is_registered_end_to_end(migrated_engine, tmp_path):
    from evolucio.cli.run_command import run_config

    registry = PostgresRegistry(create_session_factory(migrated_engine))
    experiment = registry.create_experiment(name="end-to-end")
    output = run_config(
        Path("tests/fixtures/config/valid_v1.json"),
        steps=1,
        chunk_steps=1,
        quiet=False,
        dry_run=False,
        output_root=tmp_path,
        experiment_id=experiment.experiment_id,
    )
    # text_fields JSON-encodes scalar values, including quotes around strings.
    assert 'status: "completed"' in output.splitlines()
    runs = registry.list_runs(experiment_id=experiment.experiment_id)
    assert len(runs) == 1 and runs[0].status == "completed"
    assert {record.dataset_kind for record in registry.list_datasets(runs[0].run_id)} == {
        "metrics",
        "births",
        "deaths",
        "genealogy",
    }


def test_upgrade_downgrade_upgrade(database_url):
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    command.upgrade(config, "head")


def test_verified_indexes_reconcile_and_sync_idempotently(migrated_engine, tmp_path):
    from sqlalchemy import delete, update

    from evolucio.cli.run_command import run_config
    from evolucio.persistence.postgres import (
        ArtifactIndexConflictError,
        reconcile_run,
        sync_run_artifacts,
    )
    from evolucio.persistence.postgres.models import ArtifactModel, RunSummaryModel
    from evolucio.persistence.zarr.index import load_index
    from evolucio.persistence.zarr.schemas import ZarrCapturePolicy

    sessions = create_session_factory(migrated_engine)
    registry = PostgresRegistry(sessions)
    experiment = registry.create_experiment(name="captures")
    run_config(
        Path("tests/fixtures/config/valid_v1.json"),
        steps=1,
        chunk_steps=1,
        quiet=True,
        dry_run=False,
        output_root=tmp_path,
        experiment_id=experiment.experiment_id,
        capture_policy=ZarrCapturePolicy(
            checkpoint_on_completion=True, snapshot_on_completion=True
        ),
    )
    (run,) = registry.list_runs()
    run_dir = tmp_path / str(run.run_id)
    assert registry.get_config_snapshot(run.run_id) is not None
    assert registry.get_run_summary(run.run_id) is not None
    captures = load_index(run_dir)
    assert {
        (c.capture_kind, c.state_step, c.relative_path, c.sha256)
        for c in registry.list_state_captures(run.run_id)
    } == {
        (c.kind, c.state_step, c.relative_path, c.sha256)
        for c in (*captures.checkpoints, *captures.snapshots)
    }
    assert len(registry.list_state_captures(run.run_id)) == 2
    assert reconcile_run(run_dir, registry).ok
    artifact_ids = {a.artifact_id for a in registry.list_artifacts(run.run_id)}
    sync_run_artifacts(run_dir, registry)
    sync_run_artifacts(run_dir, registry)
    assert {a.artifact_id for a in registry.list_artifacts(run.run_id)} == artifact_ids

    with sessions.begin() as session:
        session.execute(delete(ArtifactModel).where(ArtifactModel.relative_path == "summary.json"))
    assert reconcile_run(run_dir, registry).missing_in_database == ("summary.json",)
    sync_run_artifacts(run_dir, registry)
    with sessions.begin() as session:
        session.execute(
            update(ArtifactModel)
            .where(ArtifactModel.relative_path == "summary.json")
            .values(sha256="0" * 64)
        )
        session.execute(update(RunSummaryModel).values(alive_count=-1))
    report = reconcile_run(run_dir, registry)
    assert report.hash_mismatches == ("summary.json",)
    assert report.metadata_mismatches == ("run_summary",)
    with pytest.raises(ArtifactIndexConflictError):
        sync_run_artifacts(run_dir, registry)


@pytest.mark.parametrize("fail_at", ["initialization", "completion"])
def test_recorded_failure_keeps_verified_local_outputs(
    migrated_engine, tmp_path, monkeypatch, fail_at
):
    from evolucio.cli.run_command import run_config
    from evolucio.persistence.local.manifest import load_run_manifest
    from evolucio.persistence.postgres import reconcile_run
    from evolucio.runtime import initialization

    registry = PostgresRegistry(create_session_factory(migrated_engine))
    experiment = registry.create_experiment(name="failed run")

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected failure")

    if fail_at == "initialization":
        monkeypatch.setattr(initialization, "initialize_simulation", fail)
    else:
        monkeypatch.setattr(PostgresRegistry, "mark_run_completed", fail)

    with pytest.raises(RuntimeError, match="injected failure"):
        run_config(
            Path("tests/fixtures/config/valid_v1.json"),
            steps=1,
            chunk_steps=1,
            quiet=True,
            dry_run=False,
            output_root=tmp_path,
            experiment_id=experiment.experiment_id,
        )
    (run,) = registry.list_runs()
    run_dir = tmp_path / str(run.run_id)
    manifest = load_run_manifest(run_dir / "manifest.json")
    assert run.status == manifest.status == "failed"
    assert run.error_type == "RuntimeError"
    assert run.error_message == "injected failure"
    assert registry.get_config_snapshot(run.run_id) is not None
    assert registry.get_run_summary(run.run_id) is None
    assert {a.relative_path for a in registry.list_artifacts(run.run_id)} >= {
        "config.json",
        "error.json",
    }
    assert reconcile_run(run_dir, registry).ok


def test_batch_cli_groups_paired_cases(migrated_engine, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from evolucio.cli.main import app
    from evolucio.runtime import initialization

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("stop after registration")

    monkeypatch.setattr(initialization, "initialize_simulation", fail)
    registry = PostgresRegistry(create_session_factory(migrated_engine))
    experiment = registry.create_experiment(name="paired batches")
    config = "tests/fixtures/config/valid_v1.json"
    result = CliRunner().invoke(
        app,
        [
            "batch",
            "run",
            "--config",
            f"baseline={config}",
            "--config",
            f"variant={config}",
            "--seeds",
            "1,2,3",
            "--steps",
            "1",
            "--chunk-steps",
            "1",
            "--output-root",
            str(tmp_path),
            "--experiment-id",
            str(experiment.experiment_id),
        ],
    )
    assert result.exit_code == 1
    rows = registry.list_runs_for_comparison(experiment.experiment_id)
    assert [(r.case_id, r.seed) for r in rows] == [
        (case, seed) for case in ("baseline", "variant") for seed in (1, 2, 3)
    ]
    assert len({r.run_id for r in rows}) == 6
    assert len(registry.list_runs(config_hash=rows[0].config_hash, seed=1)) == 2
    assert sorted(r.batch_ordinal for r in registry.list_runs()) == list(range(1, 7))


def test_completion_rolls_back_index_inserts_on_conflict(migrated_engine, tmp_path):
    from sqlalchemy import select

    from evolucio.persistence.local import ArtifactRecord
    from evolucio.persistence.postgres import ArtifactIndexConflictError, RunSummaryRecord
    from evolucio.persistence.postgres.models import ArtifactModel
    from evolucio.persistence.postgres.types import RunIndexes

    recorder = _local_recorder(tmp_path)
    (config,) = recorder.manifest.artifacts
    sessions = create_session_factory(migrated_engine)
    registry = PostgresRegistry(sessions)
    run = registry.create_run(
        manifest=recorder.manifest,
        config_json=json.loads((recorder.run_dir / "config.json").read_text()),
        config_schema_version=3,
        config_artifact_sha256=config.sha256,
    )
    registry.mark_run_running(run.run_id)
    first = ArtifactRecord("first", "test", "first.json", "application/json", 0, "1" * 64)
    second = ArtifactRecord("second", "test", "second.json", "application/json", 0, "2" * 64)
    registry.index_artifact(run.run_id, second)
    conflict = ArtifactRecord(
        second.name,
        second.kind,
        second.relative_path,
        second.media_type,
        second.byte_size,
        "0" * 64,
    )
    with pytest.raises(ArtifactIndexConflictError):
        registry.mark_run_completed(
            run.run_id,
            summary=RunSummaryRecord(1, 0, 0, 1, 0, 0),
            final_step=4,
            steps_executed=4,
            indexes=RunIndexes(run.config_hash, (first, conflict), (), ()),
        )
    with sessions() as session:
        (persisted,) = session.scalars(select(ArtifactModel)).all()
        assert persisted.relative_path == second.relative_path
        assert persisted.sha256 == second.sha256
    assert registry.get_run(run.run_id).status == "running"
    assert registry.get_run_summary(run.run_id) is None
