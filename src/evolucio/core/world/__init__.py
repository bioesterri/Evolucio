"""Public world-initialisation API."""

from .bounds import positions_in_bounds
from .environment import (
    BASELINE_REGENERATION_MULTIPLIER,
    NO_ACTIVE_PHASE,
    EnvironmentControl,
    resolve_environment_control,
    update_environment_layer,
)
from .init import initialize_environment, initialize_resources, initialize_world
from .resources import regenerate_resources
from .update import update_world_for_step

__all__ = [
    "BASELINE_REGENERATION_MULTIPLIER",
    "NO_ACTIVE_PHASE",
    "EnvironmentControl",
    "initialize_environment",
    "initialize_resources",
    "initialize_world",
    "positions_in_bounds",
    "regenerate_resources",
    "resolve_environment_control",
    "update_environment_layer",
    "update_world_for_step",
]
