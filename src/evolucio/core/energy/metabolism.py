"""Pure pre-action basal metabolism and age advancement."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.dtypes import COUNT_DTYPE, MASK_DTYPE, REAL_DTYPE
from evolucio.core.state import PopulationState
from evolucio.core.types import Array


class PreActionMetabolismResult(eqx.Module):
    """Population and fixed-shape accounting for the pre-action phase."""

    population: PopulationState
    basal_cost_applied: Array
    age_incremented: Array
    invalid_active_energy_count: Array
    invalid_active_age_count: Array


def _energy_field(state: PopulationState) -> Array:
    return state.energy


def _age_field(state: PopulationState) -> Array:
    return state.age


def apply_basal_metabolism_and_age(
    population: PopulationState, *, basal_metabolic_cost: Array
) -> PreActionMetabolismResult:
    """Debit finite live energy and increment valid live int32 ages."""
    valid_energy = population.alive & jnp.isfinite(population.energy)
    maximum_age = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=population.age.dtype)
    valid_age = population.alive & (population.age >= 0) & (population.age < maximum_age)
    invalid_energy = population.alive & ~jnp.isfinite(population.energy)
    invalid_age = population.alive & ~((population.age >= 0) & (population.age < maximum_age))

    basal_cost = jnp.where(
        valid_energy, basal_metabolic_cost, jnp.asarray(0, dtype=REAL_DTYPE)
    ).astype(REAL_DTYPE)
    energy_after = jnp.where(valid_energy, population.energy - basal_cost, population.energy)
    age_after = jnp.where(
        valid_age, population.age + jnp.asarray(1, population.age.dtype), population.age
    )
    population_after = eqx.tree_at(_energy_field, population, energy_after)
    population_after = eqx.tree_at(_age_field, population_after, age_after)
    return PreActionMetabolismResult(
        population=population_after,
        basal_cost_applied=basal_cost,
        age_incremented=valid_age.astype(MASK_DTYPE),
        invalid_active_energy_count=jnp.sum(invalid_energy, dtype=COUNT_DTYPE),
        invalid_active_age_count=jnp.sum(invalid_age, dtype=COUNT_DTYPE),
    )
