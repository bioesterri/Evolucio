"""Portable tracking metadata with no dependency on the tracking SDK."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evolucio.persistence.local.artifacts import artifact_path

from .registry import TRACKING, TRACKING_PATH, identify, read_object


class SyncStatus(StrEnum):
    ACTIVE = "active"
    SYNCED = "synced"
    DEGRADED = "degraded"
    FAILED = "failed"


class TrackingState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    evolucio_run_id: str
    evolucio_experiment_id: str
    mlflow_run_id: str | None = None
    mlflow_experiment_id: str | None = None
    sync_status: SyncStatus = SyncStatus.ACTIVE
    last_error: str | None = None
    final_sync_completed: bool = False
    metrics_interval: int = Field(default=1, gt=0)
    log_small_artifacts: bool = True
    artifact_hashes: dict[str, str] = Field(default_factory=lambda: dict[str, str]())


def load_tracking_state(run_dir: Path) -> TrackingState | None:
    path = artifact_path(run_dir, TRACKING_PATH)
    if not path.exists():
        return None
    identify(read_object(path), TRACKING)
    return TrackingState.model_validate_json(path.read_bytes())
