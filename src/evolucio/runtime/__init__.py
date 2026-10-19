"""Host runtime API for exact, streamed simulation runs."""

from importlib import import_module
from typing import TYPE_CHECKING

from .types import ChunkConsumer, HostChunkBatch, RunPlan, RunRuntimeResult

if TYPE_CHECKING:
    from .dto_conversion import (
        DTOChunkConsumer,
        birth_events_to_dtos,
        death_events_to_dtos,
        host_chunk_batch_to_dto,
        metrics_to_dtos,
    )
    from .runner import run_simulation
    from .schema import (
        RUNTIME_SCHEMA_DIGEST,
        RUNTIME_SCHEMA_NAME,
        RUNTIME_SCHEMA_VERSION,
        runtime_schema_digest,
        runtime_schema_payload,
    )
    from .snapshot import simulation_state_to_snapshot


_EXPORT_MODULES = {
    **dict.fromkeys(
        (
            "DTOChunkConsumer",
            "birth_events_to_dtos",
            "death_events_to_dtos",
            "host_chunk_batch_to_dto",
            "metrics_to_dtos",
        ),
        ".dto_conversion",
    ),
    **dict.fromkeys(
        (
            "RUNTIME_SCHEMA_DIGEST",
            "RUNTIME_SCHEMA_NAME",
            "RUNTIME_SCHEMA_VERSION",
            "runtime_schema_digest",
            "runtime_schema_payload",
        ),
        ".schema",
    ),
    "run_simulation": ".runner",
    "simulation_state_to_snapshot": ".snapshot",
}


def __getattr__(name: str) -> object:
    """Keep execution plans usable without initializing a device backend."""
    if name in _EXPORT_MODULES:
        return getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RUNTIME_SCHEMA_DIGEST",
    "RUNTIME_SCHEMA_NAME",
    "RUNTIME_SCHEMA_VERSION",
    "ChunkConsumer",
    "DTOChunkConsumer",
    "HostChunkBatch",
    "RunPlan",
    "RunRuntimeResult",
    "birth_events_to_dtos",
    "death_events_to_dtos",
    "host_chunk_batch_to_dto",
    "metrics_to_dtos",
    "run_simulation",
    "runtime_schema_digest",
    "runtime_schema_payload",
    "simulation_state_to_snapshot",
]
