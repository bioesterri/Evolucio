"""Host runtime API for exact, streamed simulation runs."""

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
from .types import ChunkConsumer, HostChunkBatch, RunPlan, RunRuntimeResult

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
