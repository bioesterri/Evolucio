"""Persistent contract for the first local observation schema."""

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import IntEnum

OBSERVATION_SCHEMA_NAME = "local_cardinal_v1"
OBSERVATION_SCHEMA_VERSION = 1
OBSERVATION_SIZE = 15


class ObservationIndex(IntEnum):
    ENERGY_RELATIVE = 0
    AGE_RELATIVE = 1
    REPRODUCTION_MARGIN = 2
    CURRENT_RESOURCE = 3
    RESOURCE_NORTH = 4
    RESOURCE_SOUTH = 5
    RESOURCE_EAST = 6
    RESOURCE_WEST = 7
    AGENTS_NORTH = 8
    AGENTS_SOUTH = 9
    AGENTS_EAST = 10
    AGENTS_WEST = 11
    LOCAL_DENSITY = 12
    ENVIRONMENTAL_STRESS = 13
    MOVEMENT_BLOCKED = 14


class BlockedDirectionBit(IntEnum):
    NORTH = 1
    SOUTH = 2
    EAST = 4
    WEST = 8


@dataclass(frozen=True, slots=True)
class ObservationFieldSpec:
    index: int
    name: str
    minimum: float
    maximum: float
    category: str
    normalization: str
    source: str


OBSERVATION_FIELDS = (
    ObservationFieldSpec(
        0,
        "ENERGY_RELATIVE",
        0.0,
        1.0,
        "self",
        "clip(energy / maximum_energy, 0, 1)",
        "population.energy",
    ),
    ObservationFieldSpec(
        1, "AGE_RELATIVE", 0.0, 1.0, "self", "clip(age / maximum_age, 0, 1)", "population.age"
    ),
    ObservationFieldSpec(
        2,
        "REPRODUCTION_MARGIN",
        -1.0,
        1.0,
        "self",
        "clip((energy - reproduction_threshold) / maximum_energy, -1, 1)",
        "population.energy",
    ),
    ObservationFieldSpec(
        3,
        "CURRENT_RESOURCE",
        0.0,
        1.0,
        "resource",
        "clip(resource[y,x] / resource_capacity, 0, 1)",
        "world.resources",
    ),
    ObservationFieldSpec(
        4,
        "RESOURCE_NORTH",
        0.0,
        1.0,
        "resource",
        "clip(sum(north ray) / (resource_capacity * radius), 0, 1)",
        "world.resources",
    ),
    ObservationFieldSpec(
        5,
        "RESOURCE_SOUTH",
        0.0,
        1.0,
        "resource",
        "clip(sum(south ray) / (resource_capacity * radius), 0, 1)",
        "world.resources",
    ),
    ObservationFieldSpec(
        6,
        "RESOURCE_EAST",
        0.0,
        1.0,
        "resource",
        "clip(sum(east ray) / (resource_capacity * radius), 0, 1)",
        "world.resources",
    ),
    ObservationFieldSpec(
        7,
        "RESOURCE_WEST",
        0.0,
        1.0,
        "resource",
        "clip(sum(west ray) / (resource_capacity * radius), 0, 1)",
        "world.resources",
    ),
    ObservationFieldSpec(
        8,
        "AGENTS_NORTH",
        0.0,
        1.0,
        "agents",
        "clip(sum(north ray occupancy) / max_agents, 0, 1)",
        "world.occupancy",
    ),
    ObservationFieldSpec(
        9,
        "AGENTS_SOUTH",
        0.0,
        1.0,
        "agents",
        "clip(sum(south ray occupancy) / max_agents, 0, 1)",
        "world.occupancy",
    ),
    ObservationFieldSpec(
        10,
        "AGENTS_EAST",
        0.0,
        1.0,
        "agents",
        "clip(sum(east ray occupancy) / max_agents, 0, 1)",
        "world.occupancy",
    ),
    ObservationFieldSpec(
        11,
        "AGENTS_WEST",
        0.0,
        1.0,
        "agents",
        "clip(sum(west ray occupancy) / max_agents, 0, 1)",
        "world.occupancy",
    ),
    ObservationFieldSpec(
        12,
        "LOCAL_DENSITY",
        0.0,
        1.0,
        "agents",
        "clip(max(local_count[y,x] - 1, 0) / max_agents, 0, 1)",
        "world.occupancy",
    ),
    ObservationFieldSpec(
        13,
        "ENVIRONMENTAL_STRESS",
        0.0,
        1.0,
        "environment",
        "clip(environment[y,x], 0, 1)",
        "world.environment",
    ),
    ObservationFieldSpec(
        14,
        "MOVEMENT_BLOCKED",
        0.0,
        1.0,
        "boundary",
        "(north*1 + south*2 + east*4 + west*8) / 15",
        "closed world bounds",
    ),
)


def observation_schema_payload() -> dict[str, object]:
    """Return the complete JSON-compatible persistent schema descriptor."""
    return {
        "name": OBSERVATION_SCHEMA_NAME,
        "version": OBSERVATION_SCHEMA_VERSION,
        "size": OBSERVATION_SIZE,
        "fields": [asdict(field) for field in OBSERVATION_FIELDS],
        "coordinates": "position[...,0]=x; position[...,1]=y; map[y,x]",
        "cardinal_rays": {
            "order": ["north", "south", "east", "west"],
            "offsets": [[0, -1], [0, 1], [1, 0], [-1, 0]],
            "cells": (
                "distances 1..perception_radius; zero outside bounds; no diagonals or wrapping"
            ),
        },
        "focal_exclusion": "local density subtracts exactly one for an alive in-bounds focal agent",
        "blocked_bits": {"north": 1, "south": 2, "east": 4, "west": 8, "normalizer": 15},
    }


def observation_schema_digest() -> str:
    """Return the canonical SHA-256 digest of the schema payload."""
    canonical = json.dumps(
        observation_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


OBSERVATION_SCHEMA_DIGEST = "14bd8098500e8537b6144fa92d0e6b08f1a7d30340283411890586daa4781515"
