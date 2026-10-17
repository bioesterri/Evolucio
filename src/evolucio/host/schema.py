"""Versioned data boundary, independent of the simulation and its hashes."""

import hashlib
import json
from dataclasses import fields

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

HOST_DTO_SCHEMA_NAME = "host_metrics_events_snapshots_v1"
HOST_DTO_SCHEMA_VERSION = 1


def host_dto_schema_payload() -> dict[str, object]:
    """Describe scalar fields, compaction, ordering and analytical snapshots."""
    return {
        "name": HOST_DTO_SCHEMA_NAME,
        "version": HOST_DTO_SCHEMA_VERSION,
        "fields": {
            dto.__name__: [field.name for field in fields(dto)]
            for dto in (
                StepMetricsDTO,
                BirthEventDTO,
                DeathEventDTO,
                EventBatchDTO,
                ChunkTelemetryDTO,
                PopulationSnapshotDTO,
                WorldSnapshotDTO,
                SimulationSnapshotDTO,
            )
        },
        "metrics": {
            "scalars": "python_int_float",
            "deaths_by_cause": "tuple_int",
            "order": "consecutive_step",
        },
        "births": {"filter": "born_true_only", "order": ["birth_step", "child_agent_id"]},
        "deaths": {"filter": "died_true_only", "order": ["death_step", "agent_id"]},
        "events": "compact_tuples_without_inactive_sentinels",
        "chunk_interval": "start_step_inclusive_end_step_exclusive",
        "snapshot": {
            "population": "alive_only_sorted_by_agent_id",
            "state_step": "SimulationState.step_next_step_to_execute",
            "slot_index": "diagnostic_not_biological_identity",
            "arrays": "owned_numpy_ndarray_dtype_preserved_read_only",
            "population_shapes": "N_vectors_and_N_by_2_position",
            "world_shapes": "height_by_width",
            "explicit_only": True,
            "resumable": False,
            "full_genomes": False,
            "rng": False,
            "accumulator": False,
        },
        "in_compile_signature": False,
        "in_scientific_config_hash": False,
        "serialization": False,
    }


def host_dto_schema_digest() -> str:
    """Hash canonical UTF-8 JSON describing the DTO contract."""
    canonical = json.dumps(
        host_dto_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


HOST_DTO_SCHEMA_DIGEST = host_dto_schema_digest()
