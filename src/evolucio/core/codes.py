"""Stable categorical codes shared by core modules."""

from enum import IntEnum


class ActionCode(IntEnum):
    """Host representation of the fixed action vocabulary."""

    STAY = 0
    MOVE_NORTH = 1
    MOVE_SOUTH = 2
    MOVE_EAST = 3
    MOVE_WEST = 4
    EAT = 5
    REPRODUCE = 6


ACTION_COUNT = len(ActionCode)


class DeathCauseCode(IntEnum):
    """Host representation of death causes, without priority semantics."""

    NONE = 0
    ENERGY_DEPLETION = 1
    MAX_AGE = 2
    ENVIRONMENTAL_STRESS = 3
    COMPETITIVE_EXCLUSION = 4
    INVALID_STATE = 5


DEATH_CAUSE_COUNT = len(DeathCauseCode)


class RngStreamCode(IntEnum):
    """Stable domains for deterministic random-key derivation."""

    WORLD_INITIALIZATION = 0
    RESOURCE_INITIALIZATION = 1
    AGENT_INITIALIZATION = 2
    GENOME_INITIALIZATION = 3
    ENVIRONMENT_UPDATE = 4
    ACTION_TIE_BREAK = 5
    MOVEMENT_CONFLICT = 6
    FEEDING_CONFLICT = 7
    REPRODUCTION_CONFLICT = 8
    BIRTH_PLACEMENT = 9
    GENOME_MUTATION = 10


RNG_STREAM_COUNT = len(RngStreamCode)
