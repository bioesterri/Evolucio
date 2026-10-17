"""Host runtime API for exact, streamed simulation runs."""

from .runner import run_simulation
from .schema import (
    RUNTIME_SCHEMA_DIGEST,
    RUNTIME_SCHEMA_NAME,
    RUNTIME_SCHEMA_VERSION,
    runtime_schema_digest,
    runtime_schema_payload,
)
from .types import ChunkConsumer, HostChunkBatch, RunPlan, RunRuntimeResult

__all__ = [
    "RUNTIME_SCHEMA_DIGEST",
    "RUNTIME_SCHEMA_NAME",
    "RUNTIME_SCHEMA_VERSION",
    "ChunkConsumer",
    "HostChunkBatch",
    "RunPlan",
    "RunRuntimeResult",
    "run_simulation",
    "runtime_schema_digest",
    "runtime_schema_payload",
]
