"""Deterministic fixed-capacity founder population construction."""

from evolucio.core.state import INACTIVE_POSITION_COORDINATE

from .init import (
    INITIAL_AGE,
    INITIAL_BIRTH_STEP,
    INITIAL_GENERATION,
    PopulationInitializationResult,
    create_empty_population,
    initialize_population,
)

__all__ = [
    "INACTIVE_POSITION_COORDINATE",
    "INITIAL_AGE",
    "INITIAL_BIRTH_STEP",
    "INITIAL_GENERATION",
    "PopulationInitializationResult",
    "create_empty_population",
    "initialize_population",
]
