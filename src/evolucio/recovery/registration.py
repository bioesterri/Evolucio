"""Reuse the source run's registry and optional tracking binding."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from evolucio.persistence.local.manifest import load_run_manifest
from evolucio.persistence.parquet.index import load_index

if TYPE_CHECKING:
    from evolucio.persistence.postgres.recording import PostgresRunObserver
    from evolucio.tracking.mlflow.live import MlflowRunTracker


@contextmanager
def recovery_adapters(
    source: Path,
) -> Generator[tuple[PostgresRunObserver | None, MlflowRunTracker | None]]:
    linked = (source / "artifacts/postgres/experiment.json").exists()
    if not os.environ.get("EVOLUCIO_DATABASE_URL"):
        if linked or (source / "artifacts/mlflow/tracking.json").exists():
            raise ValueError("tracked recovery requires the source PostgreSQL registry")
        yield None, None
        return

    from evolucio.persistence.postgres.recording import PostgresRunObserver
    from evolucio.persistence.postgres.registry import (
        create_registry_engine,
        create_session_factory,
    )
    from evolucio.persistence.postgres.repository import PostgresRegistry

    engine = create_registry_engine()
    try:
        registry = PostgresRegistry(create_session_factory(engine))
        manifest = load_run_manifest(source / "manifest.json")
        run = registry.get_run(UUID(manifest.run_id))
        if run is None or run.experiment_id is None:
            if linked:
                raise ValueError("source run is absent from the configured PostgreSQL registry")
            yield None, None
            return
        if (run.status, run.seed, run.config_hash, run.final_step) != (
            manifest.status,
            manifest.seed,
            manifest.config_hash,
            manifest.final_step,
        ):
            raise ValueError("source manifest and PostgreSQL registry disagree")
        observer = PostgresRunObserver(registry, run.experiment_id)
        tracking = None
        if (source / "artifacts/mlflow/tracking.json").exists():
            from evolucio.tracking.mlflow.live import MlflowRunTracker
            from evolucio.tracking.mlflow.runs import load_tracking_state
            from evolucio.tracking.mlflow.types import MlflowTrackingPolicy

            state = load_tracking_state(source)
            assert state is not None
            canonical_interval = load_index(source).policy.metrics_interval
            interval = state.metrics_interval
            if interval % canonical_interval:
                # Log only steps that were eligible in both histories.
                from math import lcm

                interval = lcm(interval, canonical_interval)
            tracking = MlflowRunTracker(
                registry,
                run.experiment_id,
                MlflowTrackingPolicy(
                    interval, required=False, log_small_artifacts=state.log_small_artifacts
                ),
            )
        yield observer, tracking
    finally:
        engine.dispose()
