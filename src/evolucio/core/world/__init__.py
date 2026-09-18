"""Public world-initialisation API."""

from .bounds import positions_in_bounds
from .density import (
    compute_local_population_count,
    compute_local_population_density,
    occupancy_to_density,
)
from .environment import (
    BASELINE_REGENERATION_MULTIPLIER,
    NO_ACTIVE_PHASE,
    EnvironmentControl,
    resolve_environment_control,
    update_environment_layer,
)
from .init import initialize_environment, initialize_resources, initialize_world
from .occupancy import (
    OccupancyResult,
    WorldOccupancyUpdate,
    compute_occupancy,
    rebuild_world_occupancy,
)
from .resources import regenerate_resources
from .update import update_world_for_step

__all__ = [
    "BASELINE_REGENERATION_MULTIPLIER",
    "NO_ACTIVE_PHASE",
    "EnvironmentControl",
    "OccupancyResult",
    "WorldOccupancyUpdate",
    "compute_local_population_count",
    "compute_local_population_density",
    "compute_occupancy",
    "initialize_environment",
    "initialize_resources",
    "initialize_world",
    "occupancy_to_density",
    "positions_in_bounds",
    "rebuild_world_occupancy",
    "regenerate_resources",
    "resolve_environment_control",
    "update_environment_layer",
    "update_world_for_step",
]
