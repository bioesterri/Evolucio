"""Pure deterministic resource regeneration."""

from evolucio.core.types import Array


def regenerate_resources(
    resources: Array,
    *,
    resource_capacity: Array,
    regeneration_rate: Array,
    regeneration_multiplier: Array,
) -> Array:
    """Recover a fraction of each cell's deficit to local capacity."""
    effective_rate = regeneration_rate * regeneration_multiplier
    resource_gap = resource_capacity - resources
    return resources + resource_gap * effective_rate
