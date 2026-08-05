"""Stable public vocabulary for the functional simulation core."""

from importlib import import_module
from typing import Final

from .codes import (
    ACTION_COUNT,
    DEATH_CAUSE_COUNT,
    RNG_STREAM_COUNT,
    ActionCode,
    DeathCauseCode,
    RngStreamCode,
)
from .dtypes import (
    CODE_DTYPE,
    COUNT_DTYPE,
    ID_DTYPE,
    INDEX_DTYPE,
    MASK_DTYPE,
    REAL_DTYPE,
    STEP_DTYPE,
)
from .types import AgentId, Array, GenomeId, LineageId, Shape, StepIndex

_LAZY_EXPORTS: Final = {
    "FIRST_ID": "evolucio.core.ids",
    "MAX_NEXT_ID": "evolucio.core.ids",
    "NULL_ID": "evolucio.core.ids",
    "IdCounters": "evolucio.core.state",
    "PopulationState": "evolucio.core.state",
    "SimulationState": "evolucio.core.state",
    "WorldState": "evolucio.core.state",
    "PRNG_IMPLEMENTATION": "evolucio.core.rng",
    "RngState": "evolucio.core.rng",
    "create_rng_state": "evolucio.core.rng",
}

__all__ = [
    "ACTION_COUNT",
    "CODE_DTYPE",
    "COUNT_DTYPE",
    "DEATH_CAUSE_COUNT",
    "ID_DTYPE",
    "INDEX_DTYPE",
    "MASK_DTYPE",
    "REAL_DTYPE",
    "RNG_STREAM_COUNT",
    "STEP_DTYPE",
    "ActionCode",
    "AgentId",
    "Array",
    "DeathCauseCode",
    "GenomeId",
    "LineageId",
    "RngStreamCode",
    "Shape",
    "StepIndex",
]


def __getattr__(name: str) -> object:
    """Resolve later core symbols lazily when their modules are present."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
