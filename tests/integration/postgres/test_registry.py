# ruff: noqa: ANN001, ANN201, ANN202
import json
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

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
    artifact = next(item for item in recorder.manifest.artifacts if item.name == "configuration")
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
        item for item in second.manifest.artifacts if item.name == "configuration"
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
    assert "status: completed" in output
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
