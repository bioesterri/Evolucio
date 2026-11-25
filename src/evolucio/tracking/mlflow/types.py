"""Host-only tracking policy and persisted state."""

from dataclasses import dataclass

from evolucio.persistence.loaders.tracking import SyncStatus as SyncStatus
from evolucio.persistence.loaders.tracking import TrackingState as TrackingState


@dataclass(frozen=True, slots=True)
class MlflowTrackingPolicy:
    metrics_interval: int = 1
    log_small_artifacts: bool = True
    required: bool = True

    def __post_init__(self) -> None:
        if type(self.metrics_interval) is not int or self.metrics_interval <= 0:
            raise ValueError("metrics_interval must be a positive integer")


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
