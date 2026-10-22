"""Named query helpers; intentionally not a generic query builder."""

from uuid import UUID

from .models import ArtifactModel, DatasetIndexModel, RunModel, StateCaptureModel
from .repository import PostgresRegistry


def runs_for_experiment(registry: PostgresRegistry, experiment_id: UUID) -> tuple[RunModel, ...]:
    return registry.list_runs(experiment_id=experiment_id)


def runs_for_config_and_seed(
    registry: PostgresRegistry, config_hash: str, seed: int
) -> tuple[RunModel, ...]:
    return registry.list_runs(config_hash=config_hash, seed=seed)


def artifacts_for_run(registry: PostgresRegistry, run_id: UUID) -> tuple[ArtifactModel, ...]:
    return registry.list_artifacts(run_id)


def datasets_for_run(registry: PostgresRegistry, run_id: UUID) -> tuple[DatasetIndexModel, ...]:
    return registry.list_datasets(run_id)


def captures_for_run(registry: PostgresRegistry, run_id: UUID) -> tuple[StateCaptureModel, ...]:
    return registry.list_state_captures(run_id)
