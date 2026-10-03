"""Pure structural and final-state invariant checks."""

# pyright: reportUnknownMemberType=false

import jax.numpy as jnp

from evolucio.core.dtypes import COUNT_DTYPE, MASK_DTYPE
from evolucio.core.ids import FIRST_ID, NULL_ID
from evolucio.core.spatial import compute_occupancy
from evolucio.core.state import INACTIVE_POSITION_COORDINATE, SimulationState
from evolucio.core.types import Array
from evolucio.core.world import positions_in_bounds

from .codes import STATE_INVARIANT_CODE_COUNT, StateInvariantCode, StateInvariantReport


def _count(mask: Array) -> Array:
    return jnp.sum(mask, dtype=COUNT_DTYPE)


def _duplicate_count(values: Array, active: Array) -> Array:
    sentinel = jnp.asarray(jnp.iinfo(values.dtype).max, dtype=values.dtype)
    sorted_values = jnp.sort(jnp.where(active, values, sentinel))
    adjacent_duplicate = (sorted_values[1:] == sorted_values[:-1]) & (sorted_values[1:] != sentinel)
    return _count(adjacent_duplicate)


def _report(counts: Array) -> StateInvariantReport:
    counts = counts.astype(COUNT_DTYPE)
    failed = (counts > 0).astype(MASK_DTYPE)
    return StateInvariantReport(
        violation_counts=counts,
        failed=failed,
        total_violation_count=jnp.sum(counts, dtype=COUNT_DTYPE),
        ok=(~jnp.any(failed)).astype(MASK_DTYPE),
    )


def check_structural_invariants(
    *, state: SimulationState, resource_capacity: Array, width: int, height: int
) -> StateInvariantReport:
    """Detect internal inconsistencies without changing the supplied state."""
    population = state.population
    alive = population.alive
    inactive = ~alive
    in_bounds = positions_in_bounds(population.position, width=width, height=height)

    active_id_invalid = alive & (
        (population.agent_id < FIRST_ID)
        | (population.genome_id < FIRST_ID)
        | (population.lineage_id < FIRST_ID)
    )
    genealogy_invalid = alive & (
        (population.generation < 0)
        | ((population.generation == 0) & (population.parent_id != NULL_ID))
        | ((population.generation > 0) & (population.parent_id < FIRST_ID))
        | (population.parent_id == population.agent_id)
    )
    inactive_population_noncanonical = inactive & (
        (population.agent_id != NULL_ID)
        | (population.parent_id != NULL_ID)
        | (population.lineage_id != NULL_ID)
        | (population.genome_id != NULL_ID)
        | (population.generation != 0)
        | jnp.any(population.position != INACTIVE_POSITION_COORDINATE, axis=1)
        | (population.energy != 0)
        | (population.birth_step != 0)
        | (population.age != 0)
    )

    genome_nonfinite = jnp.zeros_like(alive)
    inactive_genome_nonzero = jnp.zeros_like(alive)
    for leaf in (
        state.genomes.layer1.weight,
        state.genomes.layer1.bias,
        state.genomes.layer2.weight,
        state.genomes.layer2.bias,
    ):
        axes = tuple(range(1, leaf.ndim))
        genome_nonfinite = genome_nonfinite | ~jnp.all(jnp.isfinite(leaf), axis=axes)
        inactive_genome_nonzero = inactive_genome_nonzero | jnp.any(leaf != 0, axis=axes)

    expected_occupancy = compute_occupancy(population, width=width, height=height).occupancy
    any_alive = jnp.any(alive)
    minimum_id = jnp.asarray(FIRST_ID, dtype=population.agent_id.dtype)
    maximum_agent = jnp.max(jnp.where(alive, population.agent_id, minimum_id - 1))
    maximum_genome = jnp.max(jnp.where(alive, population.genome_id, minimum_id - 1))
    maximum_lineage = jnp.max(jnp.where(alive, population.lineage_id, minimum_id - 1))
    counter_invalid = (
        (state.ids.next_agent_id < FIRST_ID)
        | (state.ids.next_genome_id < FIRST_ID)
        | (state.ids.next_lineage_id < FIRST_ID)
        | (any_alive & (state.ids.next_agent_id <= maximum_agent))
        | (any_alive & (state.ids.next_genome_id <= maximum_genome))
        | (any_alive & (state.ids.next_lineage_id <= maximum_lineage))
    )

    counts = jnp.zeros((STATE_INVARIANT_CODE_COUNT,), dtype=COUNT_DTYPE)
    counts = counts.at[StateInvariantCode.STEP_NEGATIVE].set((state.step < 0).astype(COUNT_DTYPE))
    counts = counts.at[StateInvariantCode.ACTIVE_ID_INVALID].set(_count(active_id_invalid))
    counts = counts.at[StateInvariantCode.ACTIVE_AGENT_ID_DUPLICATE].set(
        _duplicate_count(population.agent_id, alive)
    )
    counts = counts.at[StateInvariantCode.ACTIVE_GENOME_ID_DUPLICATE].set(
        _duplicate_count(population.genome_id, alive)
    )
    counts = counts.at[StateInvariantCode.ACTIVE_GENEALOGY_INVALID].set(_count(genealogy_invalid))
    counts = counts.at[StateInvariantCode.ACTIVE_POSITION_INVALID].set(_count(alive & ~in_bounds))
    counts = counts.at[StateInvariantCode.ACTIVE_ENERGY_NONFINITE].set(
        _count(alive & ~jnp.isfinite(population.energy))
    )
    counts = counts.at[StateInvariantCode.ACTIVE_AGE_INVALID].set(
        _count(alive & (population.age < 0))
    )
    counts = counts.at[StateInvariantCode.ACTIVE_GENOME_NONFINITE].set(
        _count(alive & genome_nonfinite)
    )
    counts = counts.at[StateInvariantCode.INACTIVE_POPULATION_NONCANONICAL].set(
        _count(inactive_population_noncanonical)
    )
    counts = counts.at[StateInvariantCode.INACTIVE_GENOME_NONZERO].set(
        _count(inactive & inactive_genome_nonzero)
    )
    counts = counts.at[StateInvariantCode.OCCUPANCY_NEGATIVE].set(_count(state.world.occupancy < 0))
    counts = counts.at[StateInvariantCode.OCCUPANCY_MISMATCH].set(
        _count(state.world.occupancy != expected_occupancy)
    )
    counts = counts.at[StateInvariantCode.RESOURCE_NONFINITE].set(
        _count(~jnp.isfinite(state.world.resources))
    )
    counts = counts.at[StateInvariantCode.RESOURCE_OUT_OF_RANGE].set(
        _count((state.world.resources < 0) | (state.world.resources > resource_capacity))
    )
    counts = counts.at[StateInvariantCode.ENVIRONMENT_NONFINITE].set(
        _count(~jnp.isfinite(state.world.environment))
    )
    counts = counts.at[StateInvariantCode.ID_COUNTER_INVALID].set(
        counter_invalid.astype(COUNT_DTYPE)
    )
    return _report(counts)


def check_final_state_invariants(
    *,
    state: SimulationState,
    resource_capacity: Array,
    death_energy_threshold: Array,
    maximum_age: Array,
    width: int,
    height: int,
) -> StateInvariantReport:
    """Add post-mortality viability checks to the structural report."""
    structural = check_structural_invariants(
        state=state, resource_capacity=resource_capacity, width=width, height=height
    )
    counts = structural.violation_counts
    counts = counts.at[StateInvariantCode.FINAL_ENERGY_NOT_VIABLE].set(
        _count(state.population.alive & (state.population.energy <= death_energy_threshold))
    )
    counts = counts.at[StateInvariantCode.FINAL_AGE_NOT_VIABLE].set(
        _count(state.population.alive & (state.population.age >= maximum_age))
    )
    return _report(counts)
