"""Recovery provenance is recorded with the new run and checked against its source."""

from pathlib import Path
from uuid import UUID

from sqlalchemy import Engine

from evolucio.config import load_config
from evolucio.persistence.local import create_local_run
from evolucio.persistence.local.manifest import RunStopRecord, utc_now
from evolucio.persistence.postgres.recording import PostgresRunObserver
from evolucio.persistence.postgres.registry import create_session_factory
from evolucio.persistence.postgres.repository import PostgresRegistry
from evolucio.recovery.provenance import PROVENANCE_PATH
from evolucio.runtime import RunMode, RunPlan


def test_recovery_registry(migrated_engine: Engine, tmp_path: Path) -> None:
    registry = PostgresRegistry(create_session_factory(migrated_engine))
    experiment = registry.create_experiment(name="Recovery")
    observer = PostgresRunObserver(registry, experiment.experiment_id)
    config = load_config("tests/fixtures/config/valid_v1.json")
    original = create_local_run(
        config=config, plan=RunPlan(4, 2), output_root=tmp_path, mode=RunMode.EXPERIMENT
    )
    observer.created(original)
    original.mark_running()
    observer.running(original)
    original.mark_stopped(RunStopRecord("user_interruption", utc_now(), utc_now(), True, 2, 2, 2))
    observer.stopped(original)
    recovered = create_local_run(
        config=config,
        plan=RunPlan(2, 2),
        output_root=tmp_path,
        mode=RunMode.EXPERIMENT,
        initial_step=2,
    )
    recovered.write_json_artifact(
        relative_path=PROVENANCE_PATH,
        name="recovery_provenance",
        kind="recovery_provenance",
        payload={
            "schema_version": 1,
            "source_run_id": original.run_id,
            "root_run_id": original.run_id,
            "source_checkpoint_step": 2,
            "source_checkpoint_hash": "a" * 64,
            "source_config_hash": original.manifest.config_hash,
            "recovery_depth": 1,
            "inherited_steps": 2,
            "new_steps_requested": 2,
            "target_step": 4,
            "inherited_metrics_rows": 2,
            "inherited_birth_rows": 0,
            "inherited_death_rows": 0,
            "inherited_genealogy_rows": 0,
            "source_prefix_digest": "b" * 64,
        },
    )
    observer.created(recovered)
    chain = registry.get_recovery_chain(UUID(recovered.run_id))
    assert len(chain) == 1 and chain[0].source_run_id == UUID(original.run_id)
    assert registry.list_recovery_attempts(UUID(original.run_id))[0].run_id == UUID(
        recovered.run_id
    )
    assert registry.list_stopped_runs()[0].run_id == UUID(original.run_id)
    assert registry.list_failed_runs() == ()
