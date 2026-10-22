"""Compose the canonical component initializers into a complete device state."""

import jax.numpy as jnp

from evolucio.config.compile import CoreConfig
from evolucio.core.dtypes import STEP_DTYPE
from evolucio.core.ids import create_id_counters
from evolucio.core.policy import initialize_genome_batch
from evolucio.core.population import initialize_population
from evolucio.core.rng import create_rng_state
from evolucio.core.state import SimulationState
from evolucio.core.world import initialize_world


class InitializationError(ValueError):
    """Canonical component initialization reported an invalid initial state."""


def initialize_simulation(config: CoreConfig, *, seed: int) -> SimulationState:
    """Initialize founders, occupancy and genomes with one explicit root seed."""
    rng = create_rng_state(seed)
    world = initialize_world(config.world, rng.key)
    founders = initialize_population(
        world, config.population, config.energy, rng.key, create_id_counters()
    )
    if bool(founders.overflow):
        raise InitializationError("founder identifier allocation overflow")
    genomes = initialize_genome_batch(
        alive=founders.population.alive,
        genome_ids=founders.population.genome_id,
        root_key=rng.key,
    )
    if int(genomes.invalid_active_genome_id_count):
        raise InitializationError("invalid active founder genome identity")
    return SimulationState(
        step=jnp.asarray(0, dtype=STEP_DTYPE),  # pyright: ignore[reportUnknownMemberType]
        rng=rng,
        ids=founders.ids,
        world=founders.world,
        population=founders.population,
        genomes=genomes.genomes,
    )
