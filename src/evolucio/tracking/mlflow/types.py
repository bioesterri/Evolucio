"""Host-only tracking policy and persisted state."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SyncStatus(StrEnum):
    ACTIVE = "active"
    SYNCED = "synced"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MlflowTrackingPolicy:
    metrics_interval: int = 1
    log_small_artifacts: bool = True
    required: bool = True

    def __post_init__(self) -> None:
        if type(self.metrics_interval) is not int or self.metrics_interval <= 0:
            raise ValueError("metrics_interval must be a positive integer")


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


@dataclass(frozen=True, slots=True)
class MlflowReconciliationReport:
    ok: bool
    missing_binding: tuple[str, ...] = ()
    missing_params: tuple[str, ...] = ()
    param_mismatches: tuple[str, ...] = ()
    tag_mismatches: tuple[str, ...] = ()
    metric_mismatches: tuple[str, ...] = ()
    status_mismatch: bool = False
    missing_metrics: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    mismatches: tuple[str, ...] = ()
