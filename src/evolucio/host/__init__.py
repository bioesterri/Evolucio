"""Device-independent DTOs for future persistence, analysis and visualization."""

from .dto import (
    BirthEventDTO,
    ChunkTelemetryDTO,
    DeathEventDTO,
    EventBatchDTO,
    PopulationSnapshotDTO,
    SimulationSnapshotDTO,
    StepMetricsDTO,
    WorldSnapshotDTO,
)
from .schema import (
    HOST_DTO_SCHEMA_DIGEST,
    HOST_DTO_SCHEMA_NAME,
    HOST_DTO_SCHEMA_VERSION,
    host_dto_schema_digest,
    host_dto_schema_payload,
)

__all__ = [
    "HOST_DTO_SCHEMA_DIGEST",
    "HOST_DTO_SCHEMA_NAME",
    "HOST_DTO_SCHEMA_VERSION",
    "BirthEventDTO",
    "ChunkTelemetryDTO",
    "DeathEventDTO",
    "EventBatchDTO",
    "PopulationSnapshotDTO",
    "SimulationSnapshotDTO",
    "StepMetricsDTO",
    "WorldSnapshotDTO",
    "host_dto_schema_digest",
    "host_dto_schema_payload",
]
