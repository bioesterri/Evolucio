"""Pure fixed-shape birth events and genealogy validation."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.dtypes import COUNT_DTYPE, INDEX_DTYPE, MASK_DTYPE
from evolucio.core.ids import MAX_NEXT_ID, NULL_ID
from evolucio.core.state import PopulationState
from evolucio.core.types import Array

from .mutation import GenomeMutationResult
from .reproduction import ReproductionResolutionResult


class BirthEventBatch(eqx.Module):
    """Fixed-capacity birth records aligned with population slots."""

    born: Array
    birth_step: Array
    child_agent_id: Array
    parent_agent_id: Array
    lineage_id: Array
    generation: Array
    child_genome_id: Array
    parent_genome_id: Array
    birth_position: Array
    child_initial_energy: Array
    parent_energy_before: Array
    parent_energy_after: Array
    mutation_selected_count: Array
    mutation_effective_count: Array
    mutation_selected_weight_count: Array
    mutation_selected_bias_count: Array
    mutation_sum_abs_delta: Array
    mutation_max_abs_delta: Array


class GenealogyValidationResult(eqx.Module):
    """Per-slot genealogy validity and scalar inconsistency diagnostics."""

    valid_birth: Array
    invalid_birth_count: Array
    mutation_inconsistency_count: Array


def _masked(values: Array, born: Array, fill: int | float) -> Array:
    return jnp.where(born, values, jnp.full_like(values, fill))


def build_birth_events(
    *,
    population_before_reproduction: PopulationState,
    reproduction: ReproductionResolutionResult,
    mutation: GenomeMutationResult,
    step: Array,
) -> BirthEventBatch:
    """Materialize an immutable audit trace for births already resolved and mutated."""
    born = reproduction.newborn_mask.astype(MASK_DTYPE)
    capacity = born.shape[0]
    safe_parent_slots = jnp.clip(reproduction.parent_slots, 0, capacity - 1)
    before = population_before_reproduction
    child = reproduction.population

    return BirthEventBatch(
        born=born,
        birth_step=_masked(jnp.broadcast_to(step, (capacity,)), born, 0),
        child_agent_id=_masked(child.agent_id, born, NULL_ID),
        parent_agent_id=_masked(child.parent_id, born, NULL_ID),
        lineage_id=_masked(child.lineage_id, born, NULL_ID),
        generation=_masked(child.generation, born, 0),
        child_genome_id=_masked(child.genome_id, born, NULL_ID),
        parent_genome_id=_masked(before.genome_id[safe_parent_slots], born, NULL_ID),
        birth_position=jnp.where(
            born[:, None], child.position, jnp.full_like(child.position, -1)
        ).astype(INDEX_DTYPE),
        child_initial_energy=_masked(child.energy, born, 0.0),
        parent_energy_before=_masked(before.energy[safe_parent_slots], born, 0.0),
        parent_energy_after=_masked(child.energy[safe_parent_slots], born, 0.0),
        mutation_selected_count=_masked(mutation.selected_parameter_count, born, 0),
        mutation_effective_count=_masked(mutation.effective_parameter_count, born, 0),
        mutation_selected_weight_count=_masked(mutation.selected_weight_count, born, 0),
        mutation_selected_bias_count=_masked(mutation.selected_bias_count, born, 0),
        mutation_sum_abs_delta=_masked(mutation.sum_abs_delta, born, 0.0),
        mutation_max_abs_delta=_masked(mutation.max_abs_delta, born, 0.0),
    )


def validate_genealogy(
    *,
    population_before_reproduction: PopulationState,
    reproduction: ReproductionResolutionResult,
    mutation: GenomeMutationResult,
    step: Array,
) -> GenealogyValidationResult:
    """Validate observed births without repairing or changing simulation state."""
    born = reproduction.newborn_mask
    capacity = born.shape[0]
    parent_slots = reproduction.parent_slots
    parent_slot_valid = (parent_slots >= 0) & (parent_slots < capacity)
    safe_parent_slots = jnp.clip(parent_slots, 0, capacity - 1)
    parent = population_before_reproduction
    child = reproduction.population
    parent_generation = parent.generation[safe_parent_slots]
    valid = (
        born
        & child.alive
        & (child.agent_id >= 0)
        & (child.genome_id >= 0)
        & parent_slot_valid
        & parent.alive[safe_parent_slots]
        & (parent.agent_id[safe_parent_slots] == child.parent_id)
        & (child.lineage_id == parent.lineage_id[safe_parent_slots])
        & (parent_generation < MAX_NEXT_ID)
        & (child.generation == parent_generation + 1)
        & (child.birth_step == step)
        & (child.age == 0)
        & (child.agent_id != parent.agent_id[safe_parent_slots])
        & (child.genome_id != parent.genome_id[safe_parent_slots])
    )
    mutation_activity = (
        (mutation.selected_parameter_count != 0)
        | (mutation.effective_parameter_count != 0)
        | (mutation.selected_weight_count != 0)
        | (mutation.selected_bias_count != 0)
        | (mutation.sum_abs_delta != 0)
        | (mutation.max_abs_delta != 0)
    )
    mutation_inconsistent = mutation_activity & ~born
    return GenealogyValidationResult(
        valid_birth=valid.astype(MASK_DTYPE),
        invalid_birth_count=jnp.sum(born & ~valid, dtype=COUNT_DTYPE),
        mutation_inconsistency_count=jnp.sum(mutation_inconsistent, dtype=COUNT_DTYPE),
    )
