"""Atomic fixed-capacity asexual reproduction."""

# pyright: reportUnknownMemberType=false

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.codes import ActionCode
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, ID_DTYPE, INDEX_DTYPE, MASK_DTYPE
from evolucio.core.ids import MAX_NEXT_ID, IdCounters
from evolucio.core.policy.batch import GenomeBatch
from evolucio.core.policy.model import PolicyLinear
from evolucio.core.rng import derive_entity_keys
from evolucio.core.spatial import rebuild_world_occupancy
from evolucio.core.state import PopulationState, WorldState
from evolucio.core.types import Array
from evolucio.core.viability import ReproductionGateResult

_CARDINAL_OFFSETS = jnp.asarray(((0, -1), (0, 1), (1, 0), (-1, 0)), dtype=INDEX_DTYPE)


class ReproductionResolutionCode(IntEnum):
    """Outcome of each population slot's birth attempt."""

    NOT_ELIGIBLE = 0
    BIRTH_SUCCEEDED = 1
    NO_BIRTH_POSITION = 2
    BIRTH_POSITION_CONFLICT = 3
    NO_FREE_POPULATION_SLOT = 4
    INVALID_REPRODUCTION_INPUT = 5
    ID_OVERFLOW = 6


class ReproductionResolutionResult(eqx.Module):
    """State and fixed-capacity diagnostics produced by reproduction."""

    population: PopulationState
    genomes: GenomeBatch
    world: WorldState
    ids: IdCounters
    newborn_mask: Array
    parent_slots: Array
    birth_positions: Array
    reproduction_codes: Array
    birth_count: Array
    no_birth_position_count: Array
    position_conflict_count: Array
    no_free_slot_count: Array
    invalid_input_count: Array
    id_overflow: Array


def _random_priorities(key: Array, agent_ids: Array, count: int) -> Array:
    keys = derive_entity_keys(key, agent_ids)

    def sample(item: Array) -> Array:
        return jax.random.bits(item, shape=(count,), dtype=jnp.uint32)

    return jax.vmap(sample)(keys)


def _replace_rows(original: Array, child_slots: Array, values: Array) -> Array:
    shape = (child_slots.shape[0],) + (1,) * (original.ndim - 1)
    return jnp.where(child_slots.reshape(shape), values, original)


def resolve_asexual_reproduction(
    *,
    population: PopulationState,
    genomes: GenomeBatch,
    world: WorldState,
    ids: IdCounters,
    reproduction_gate: ReproductionGateResult,
    actions_after_viability: Array,
    step: Array,
    reproduction_energy_cost: Array,
    offspring_initial_energy: Array,
    birth_placement_key: Array,
    reproduction_conflict_key: Array,
    death_energy_threshold: Array,
    width: int,
    height: int,
) -> ReproductionResolutionResult:
    """Resolve local births as one pure, deterministic atomic transaction."""
    capacity = population.alive.shape[0]
    slots = jnp.arange(capacity, dtype=INDEX_DTYPE)
    eligible = reproduction_gate.eligible
    scalar_valid = (
        jnp.isfinite(reproduction_energy_cost)
        & jnp.isfinite(offspring_initial_energy)
        & jnp.isfinite(death_energy_threshold)
        & (reproduction_energy_cost > 0)
        & (offspring_initial_energy > 0)
        & (offspring_initial_energy <= reproduction_energy_cost)
    )
    projected = population.energy - reproduction_energy_cost
    valid = (
        eligible
        & scalar_valid
        & population.alive
        & (population.agent_id >= 0)
        & (population.lineage_id >= 0)
        & (population.genome_id >= 0)
        & (actions_after_viability == int(ActionCode.REPRODUCE))
        & jnp.isfinite(population.energy)
        & (projected > death_energy_threshold)
    )
    invalid = eligible & ~valid

    candidates = population.position[:, None, :] + _CARDINAL_OFFSETS[None, :, :]
    in_bounds = (
        (candidates[:, :, 0] >= 0)
        & (candidates[:, :, 0] < width)
        & (candidates[:, :, 1] >= 0)
        & (candidates[:, :, 1] < height)
    )
    safe_x = jnp.clip(candidates[:, :, 0], 0, width - 1)
    safe_y = jnp.clip(candidates[:, :, 1], 0, height - 1)
    available = in_bounds & (world.occupancy[safe_y, safe_x] == 0)
    placement_priority = _random_priorities(birth_placement_key, population.agent_id, 4)
    scored = jnp.where(available, placement_priority, jnp.asarray(0, dtype=jnp.uint32))
    direction = jnp.argmax(scored, axis=1).astype(INDEX_DTYPE)
    has_position = jnp.any(available, axis=1)
    chosen_position = candidates[slots, direction]
    spatial_candidate = valid & has_position

    conflict_priority = _random_priorities(reproduction_conflict_key, population.agent_id, 2)
    same_cell = jnp.all(chosen_position[:, None, :] == chosen_position[None, :, :], axis=2)
    higher = (conflict_priority[:, 0, None] > conflict_priority[None, :, 0]) | (
        (conflict_priority[:, 0, None] == conflict_priority[None, :, 0])
        & (population.agent_id[:, None] > population.agent_id[None, :])
    )
    defeated = jnp.any(same_cell & spatial_candidate[None, :] & higher.T, axis=1)
    spatial_winner = spatial_candidate & ~defeated

    free_slots = ~population.alive
    free_count = jnp.sum(free_slots, dtype=COUNT_DTYPE)
    capacity_higher = (conflict_priority[:, 1, None] > conflict_priority[None, :, 1]) | (
        (conflict_priority[:, 1, None] == conflict_priority[None, :, 1])
        & (population.agent_id[:, None] > population.agent_id[None, :])
    )
    priority_rank = jnp.sum(capacity_higher.T & spatial_winner[None, :], axis=1, dtype=COUNT_DTYPE)
    accepted = spatial_winner & (priority_rank < free_count)
    birth_count_before_ids = jnp.sum(accepted, dtype=COUNT_DTYPE)

    remaining_agent = jnp.asarray(MAX_NEXT_ID, dtype=ID_DTYPE) - ids.next_agent_id
    remaining_genome = jnp.asarray(MAX_NEXT_ID, dtype=ID_DTYPE) - ids.next_genome_id
    overflow = (birth_count_before_ids > remaining_agent) | (
        birth_count_before_ids > remaining_genome
    )
    committed = accepted & ~overflow
    birth_count = jnp.sum(committed, dtype=COUNT_DTYPE)

    identity_before = accepted[None, :] & (
        population.agent_id[None, :] < population.agent_id[:, None]
    )
    identity_rank = jnp.sum(identity_before, axis=1, dtype=ID_DTYPE)
    new_agent_by_parent = ids.next_agent_id + identity_rank
    new_genome_by_parent = ids.next_genome_id + identity_rank

    accepted_order = jnp.sum(accepted[None, :] & (slots[None, :] < slots[:, None]), axis=1)
    free_order = jnp.sum(free_slots[None, :] & (slots[None, :] < slots[:, None]), axis=1)
    assignment = (
        free_slots[:, None] & accepted[None, :] & (free_order[:, None] == accepted_order[None, :])
    )
    parent_for_child = jnp.argmax(assignment, axis=1).astype(INDEX_DTYPE)
    child_slots = jnp.any(assignment, axis=1) & ~overflow
    safe_parent = parent_for_child

    parent_slots = jnp.where(child_slots, safe_parent, -1).astype(INDEX_DTYPE)
    birth_positions = jnp.where(
        child_slots[:, None], chosen_position[safe_parent], jnp.asarray(-1, dtype=INDEX_DTYPE)
    ).astype(INDEX_DTYPE)
    child_agent_ids = new_agent_by_parent[safe_parent]
    child_genome_ids = new_genome_by_parent[safe_parent]

    parent_paid = committed
    energy = jnp.where(parent_paid, projected, population.energy)
    updated_population = PopulationState(
        alive=jnp.where(child_slots, True, population.alive).astype(MASK_DTYPE),
        agent_id=_replace_rows(population.agent_id, child_slots, child_agent_ids),
        parent_id=_replace_rows(
            population.parent_id, child_slots, population.agent_id[safe_parent]
        ),
        lineage_id=_replace_rows(
            population.lineage_id, child_slots, population.lineage_id[safe_parent]
        ),
        genome_id=_replace_rows(population.genome_id, child_slots, child_genome_ids),
        generation=_replace_rows(
            population.generation, child_slots, population.generation[safe_parent] + 1
        ),
        position=_replace_rows(population.position, child_slots, birth_positions),
        energy=_replace_rows(
            energy, child_slots, jnp.broadcast_to(offspring_initial_energy, (capacity,))
        ),
        birth_step=_replace_rows(
            population.birth_step, child_slots, jnp.broadcast_to(step, (capacity,))
        ),
        age=_replace_rows(population.age, child_slots, jnp.zeros_like(population.age)),
    )
    updated_genomes = GenomeBatch(
        layer1=PolicyLinear(
            weight=_replace_rows(
                genomes.layer1.weight, child_slots, genomes.layer1.weight[safe_parent]
            ),
            bias=_replace_rows(genomes.layer1.bias, child_slots, genomes.layer1.bias[safe_parent]),
        ),
        layer2=PolicyLinear(
            weight=_replace_rows(
                genomes.layer2.weight, child_slots, genomes.layer2.weight[safe_parent]
            ),
            bias=_replace_rows(genomes.layer2.bias, child_slots, genomes.layer2.bias[safe_parent]),
        ),
    )
    updated_ids = IdCounters(
        next_agent_id=ids.next_agent_id + birth_count,
        next_genome_id=ids.next_genome_id + birth_count,
        next_lineage_id=ids.next_lineage_id,
    )
    rebuilt_world = rebuild_world_occupancy(
        world, updated_population, width=width, height=height
    ).world
    updated_world = WorldState(
        resources=world.resources,
        environment=world.environment,
        occupancy=jnp.where(birth_count > 0, rebuilt_world.occupancy, world.occupancy),
    )

    no_position = valid & ~has_position
    conflict = spatial_candidate & defeated
    no_slot = spatial_winner & ~accepted
    codes = jnp.full((capacity,), int(ReproductionResolutionCode.NOT_ELIGIBLE), dtype=CODE_DTYPE)
    codes = jnp.where(invalid, int(ReproductionResolutionCode.INVALID_REPRODUCTION_INPUT), codes)
    codes = jnp.where(no_position, int(ReproductionResolutionCode.NO_BIRTH_POSITION), codes)
    codes = jnp.where(conflict, int(ReproductionResolutionCode.BIRTH_POSITION_CONFLICT), codes)
    codes = jnp.where(no_slot, int(ReproductionResolutionCode.NO_FREE_POPULATION_SLOT), codes)
    codes = jnp.where(committed, int(ReproductionResolutionCode.BIRTH_SUCCEEDED), codes)
    codes = jnp.where(accepted & overflow, int(ReproductionResolutionCode.ID_OVERFLOW), codes)
    return ReproductionResolutionResult(
        population=updated_population,
        genomes=updated_genomes,
        world=updated_world,
        ids=updated_ids,
        newborn_mask=child_slots,
        parent_slots=parent_slots,
        birth_positions=birth_positions,
        reproduction_codes=codes,
        birth_count=birth_count,
        no_birth_position_count=jnp.sum(no_position, dtype=COUNT_DTYPE),
        position_conflict_count=jnp.sum(conflict, dtype=COUNT_DTYPE),
        no_free_slot_count=jnp.sum(no_slot, dtype=COUNT_DTYPE),
        invalid_input_count=jnp.sum(invalid, dtype=COUNT_DTYPE),
        id_overflow=overflow.astype(MASK_DTYPE),
    )
