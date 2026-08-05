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
    """Host representation of deterministic RNG stream identifiers."""

    INITIALIZATION = 0
    ENVIRONMENT = 1
    ACTION_CONFLICT = 2
    REPRODUCTION = 3
    MUTATION = 4


RNG_STREAM_COUNT = len(RngStreamCode)
