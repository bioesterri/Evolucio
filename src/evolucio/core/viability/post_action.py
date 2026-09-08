"""Pure, vectorised post-action viability resolution."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.codes import ActionCode
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE
from evolucio.core.policy.batch import GenomeBatch
from evolucio.core.spatial import rebuild_world_occupancy
from evolucio.core.state import PopulationState, WorldState
from evolucio.core.types import Array

from .death import (
    DeathRecordBatch,
    build_death_records,
    canonicalize_genomes,
    canonicalize_population,
    evaluate_deaths,
)


class PostActionViabilityResult(eqx.Module):
    """State and diagnostics after the second viability barrier."""

    population: PopulationState
    genomes: GenomeBatch
    world: WorldState
    deaths: DeathRecordBatch
    actions_after_viability: Array
    death_count: Array
    invalid_state_death_count: Array
    energy_death_count: Array
    max_age_death_count: Array


def resolve_post_action_viability(
    *,
    population: PopulationState,
    genomes: GenomeBatch,
    world: WorldState,
    actions_after_feeding: Array,
    step: Array,
    death_energy_threshold: Array,
    maximum_age: Array,
    width: int,
    height: int,
) -> PostActionViabilityResult:
    """Remove agents made non-viable by action gains and costs."""
    evaluation = evaluate_deaths(
        population=population,
        step=step,
        death_energy_threshold=death_energy_threshold,
        maximum_age=maximum_age,
        width=width,
        height=height,
    )
    deaths = build_death_records(
        population=population, dies=evaluation.dies, cause=evaluation.cause, step=step
    )
    population_after = canonicalize_population(population, evaluation.dies)
    genomes_after = canonicalize_genomes(genomes, evaluation.dies)
    occupancy_update = rebuild_world_occupancy(world, population_after, width=width, height=height)
    actions_after_viability = jnp.where(
        evaluation.dies,
        jnp.asarray(ActionCode.STAY, dtype=CODE_DTYPE),
        actions_after_feeding,
    ).astype(CODE_DTYPE)
    return PostActionViabilityResult(
        population=population_after,
        genomes=genomes_after,
        world=occupancy_update.world,
        deaths=deaths,
        actions_after_viability=actions_after_viability,
        death_count=jnp.sum(evaluation.dies, dtype=COUNT_DTYPE),
        invalid_state_death_count=jnp.sum(evaluation.invalid_deaths, dtype=COUNT_DTYPE),
        energy_death_count=jnp.sum(evaluation.energy_deaths, dtype=COUNT_DTYPE),
        max_age_death_count=jnp.sum(evaluation.max_age_deaths, dtype=COUNT_DTYPE),
    )
