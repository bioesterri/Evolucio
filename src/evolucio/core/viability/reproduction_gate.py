"""Projected-survival gate for post-action reproduction requests."""

# pyright: reportUnknownMemberType=false

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.codes import ActionCode
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, REAL_DTYPE
from evolucio.core.state import PopulationState
from evolucio.core.types import Array


class ReproductionGateCode(IntEnum):
    """Exclusive outcome codes in host representation."""

    NOT_REQUESTED = 0
    ELIGIBLE = 1
    DEAD_POST_ACTION = 2
    BELOW_ENERGY_THRESHOLD = 3
    BELOW_MINIMUM_AGE = 4
    SUICIDAL_PROJECTED_ENERGY = 5
    INVALID_REPRODUCTION_INPUT = 6


class ReproductionGateResult(eqx.Module):
    """Fixed-capacity reproduction candidates and diagnostics."""

    eligible: Array
    gate_codes: Array
    projected_parent_energy: Array
    requested_count: Array
    eligible_count: Array
    suicidal_block_count: Array


def evaluate_reproduction_gate(
    *,
    population_before_viability: PopulationState,
    population_after_viability: PopulationState,
    actions_before_viability: Array,
    actions_after_viability: Array,
    reproduction_energy_threshold: Array,
    minimum_reproduction_age: Array,
    reproduction_energy_cost: Array,
    offspring_initial_energy: Array,
    death_energy_threshold: Array,
) -> ReproductionGateResult:
    """Select requests whose parent would remain strictly viable after birth cost."""
    requested = population_before_viability.alive & (
        actions_before_viability == int(ActionCode.REPRODUCE)
    )
    dead_post_action = requested & ~population_after_viability.alive
    surviving_request = requested & population_after_viability.alive
    inputs_valid = (
        jnp.isfinite(reproduction_energy_threshold)
        & jnp.isfinite(reproduction_energy_cost)
        & jnp.isfinite(offspring_initial_energy)
        & jnp.isfinite(death_energy_threshold)
        & (reproduction_energy_threshold > death_energy_threshold)
        & (reproduction_energy_cost > 0)
        & (offspring_initial_energy > death_energy_threshold)
        & (offspring_initial_energy <= reproduction_energy_cost)
        & (minimum_reproduction_age >= 0)
        & jnp.isfinite(population_after_viability.energy)
    )
    invalid_input = surviving_request & (
        ~inputs_valid | (actions_after_viability != int(ActionCode.REPRODUCE))
    )
    valid_request = surviving_request & ~invalid_input
    below_energy = valid_request & (
        population_after_viability.energy < reproduction_energy_threshold
    )
    below_age = (
        valid_request & ~below_energy & (population_after_viability.age < minimum_reproduction_age)
    )
    projected = population_after_viability.energy - reproduction_energy_cost
    suicidal = valid_request & ~below_energy & ~below_age & (projected <= death_energy_threshold)
    eligible = valid_request & ~below_energy & ~below_age & ~suicidal

    # Assign from lowest to highest precedence so earlier contract rules win.
    gate_codes = jnp.full(
        requested.shape, int(ReproductionGateCode.NOT_REQUESTED), dtype=CODE_DTYPE
    )
    gate_codes = jnp.where(eligible, int(ReproductionGateCode.ELIGIBLE), gate_codes)
    gate_codes = jnp.where(
        suicidal, int(ReproductionGateCode.SUICIDAL_PROJECTED_ENERGY), gate_codes
    )
    gate_codes = jnp.where(below_age, int(ReproductionGateCode.BELOW_MINIMUM_AGE), gate_codes)
    gate_codes = jnp.where(
        below_energy, int(ReproductionGateCode.BELOW_ENERGY_THRESHOLD), gate_codes
    )
    gate_codes = jnp.where(
        invalid_input, int(ReproductionGateCode.INVALID_REPRODUCTION_INPUT), gate_codes
    )
    gate_codes = jnp.where(
        dead_post_action, int(ReproductionGateCode.DEAD_POST_ACTION), gate_codes
    ).astype(CODE_DTYPE)
    projected_parent_energy = jnp.where(
        valid_request,
        projected,
        jnp.asarray(0, dtype=REAL_DTYPE),
    ).astype(REAL_DTYPE)
    return ReproductionGateResult(
        eligible=eligible,
        gate_codes=gate_codes,
        projected_parent_energy=projected_parent_energy,
        requested_count=jnp.sum(requested, dtype=COUNT_DTYPE),
        eligible_count=jnp.sum(eligible, dtype=COUNT_DTYPE),
        suicidal_block_count=jnp.sum(suicidal, dtype=COUNT_DTYPE),
    )
