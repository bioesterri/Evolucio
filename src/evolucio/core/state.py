"""Fixed-shape PyTree containers for the simulation state."""

import equinox as eqx

from .ids import IdCounters
from .rng import RngState
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
    """Complete fixed-shape state contract currently owned by the core."""

    step: Array
    rng: RngState
    ids: IdCounters
    world: WorldState
    population: PopulationState
