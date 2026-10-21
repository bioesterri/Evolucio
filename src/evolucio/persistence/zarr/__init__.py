"""Local checkpoints and analytical snapshots; device APIs are imported explicitly."""

from importlib import import_module
from typing import TYPE_CHECKING

from .schemas import ZarrCapturePolicy

if TYPE_CHECKING:
    from .checkpoint import CheckpointRestore, load_checkpoint, write_checkpoint
    from .index import CheckpointRecord, SnapshotRecord, latest_checkpoint, list_snapshots
    from .integrity import ZarrArtifactVerification, verify_zarr_artifacts
    from .snapshot import load_snapshot, write_snapshot
    from .storage import CheckpointCompatibilityError, CheckpointIntegrityError

_MODULES = {
    "CheckpointRestore": ".checkpoint",
    "load_checkpoint": ".checkpoint",
    "write_checkpoint": ".checkpoint",
    "CheckpointRecord": ".index",
    "SnapshotRecord": ".index",
    "latest_checkpoint": ".index",
    "list_snapshots": ".index",
    "ZarrArtifactVerification": ".integrity",
    "verify_zarr_artifacts": ".integrity",
    "load_snapshot": ".snapshot",
    "write_snapshot": ".snapshot",
    "CheckpointCompatibilityError": ".storage",
    "CheckpointIntegrityError": ".storage",
}


def __getattr__(name: str) -> object:
    if name in _MODULES:
        return getattr(import_module(_MODULES[name], __name__), name)
    raise AttributeError(name)


__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointIntegrityError",
    "CheckpointRecord",
    "CheckpointRestore",
    "SnapshotRecord",
    "ZarrArtifactVerification",
    "ZarrCapturePolicy",
    "latest_checkpoint",
    "list_snapshots",
    "load_checkpoint",
    "load_snapshot",
    "verify_zarr_artifacts",
    "write_checkpoint",
    "write_snapshot",
]
