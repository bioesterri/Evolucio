"""Fixed-shape PyTree containers for the simulation state."""

import equinox as eqx

from .types import Array


class WorldState(eqx.Module):
    """Spatial fields with fixed world dimensions."""

    resources: Array
    environment: Array
    occupancy: Array


class PopulationState(eqx.Module):
    """Population data stored as fixed-capacity arrays."""

    alive: Array
    agent_id: Array
    parent_id: Array
    lineage_id: Array
    genome_id: Array
    generation: Array
    position: Array
    energy: Array
    birth_step: Array
    age: Array


class SimulationState(eqx.Module):
    """First fixed-shape state contract for the simulation core."""

    step: Array
    world: WorldState
    population: PopulationState
