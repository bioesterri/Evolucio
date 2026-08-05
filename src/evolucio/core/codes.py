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
