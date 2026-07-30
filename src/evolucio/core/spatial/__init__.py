"""Public spatial map operations for the fixed-capacity simulation core."""

from .density import (
    compute_local_population_count,
    compute_local_population_density,
    occupancy_to_density,
)
from .gather import gather_map_values
from .occupancy import (
    OccupancyResult,
    WorldOccupancyUpdate,
    compute_occupancy,
    rebuild_world_occupancy,
)

__all__ = [
    "OccupancyResult",
    "WorldOccupancyUpdate",
    "compute_local_population_count",
    "compute_local_population_density",
    "compute_occupancy",
    "gather_map_values",
    "occupancy_to_density",
    "rebuild_world_occupancy",
]
