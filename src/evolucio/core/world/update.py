"""Temporal world preparation performed before decisions for a step."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import TYPE_CHECKING

from evolucio.core.state import WorldState
from evolucio.core.types import Array

from .environment import resolve_environment_control, update_environment_layer
from .resources import regenerate_resources

if TYPE_CHECKING:
    from evolucio.config.compile import WorldCoreConfig


def update_world_for_step(world: WorldState, step: Array, config: WorldCoreConfig) -> WorldState:
    """Apply environment controls and regenerate resources before decisions."""
    control = resolve_environment_control(step, config)

    def regenerate(resources: Array) -> Array:
        return regenerate_resources(
            resources,
            resource_capacity=config.resource_capacity,
            regeneration_rate=config.regeneration_rate,
            regeneration_multiplier=control.regeneration_multiplier,
        )

    resources = regenerate(world.resources)
    return WorldState(
        resources=resources,
        environment=update_environment_layer(world.environment, control.environment_value),
        occupancy=world.occupancy,
    )
