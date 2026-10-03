from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from evolucio.core.actions import FeedingResolutionResult, MovementResolutionResult
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, ID_DTYPE, INDEX_DTYPE, REAL_DTYPE
from evolucio.core.energy import ActionEnergyCostResult, PreActionMetabolismResult
from evolucio.core.evolution import BirthEventBatch
from evolucio.core.policy import create_empty_genome_batch
from evolucio.core.state import PopulationState, WorldState
from evolucio.core.viability import build_death_records


def population(
    alive: list[bool], *, energy: list[float] | None = None, age: list[int] | None = None
) -> PopulationState:
    capacity = len(alive)
    active_ids = jnp.where(jnp.asarray(alive), jnp.arange(capacity), -1).astype(ID_DTYPE)
    zeros = jnp.zeros((capacity,), dtype=COUNT_DTYPE)
    return PopulationState(
        alive=jnp.asarray(alive),
        agent_id=active_ids,
        parent_id=jnp.full((capacity,), -1, dtype=ID_DTYPE),
        lineage_id=jnp.where(jnp.asarray(alive), jnp.asarray([3, 3, 8, 9][:capacity]), -1),
        genome_id=active_ids,
        generation=zeros,
        position=jnp.zeros((capacity, 2), dtype=INDEX_DTYPE),
        energy=jnp.asarray(energy or [0.0] * capacity, dtype=REAL_DTYPE),
        birth_step=zeros,
        age=jnp.asarray(age or [0] * capacity, dtype=COUNT_DTYPE),
    )


def births(capacity: int, born_slots: tuple[int, ...] = ()) -> BirthEventBatch:
    born = jnp.zeros((capacity,), dtype=jnp.bool_).at[jnp.asarray(born_slots)].set(True)
    ids = jnp.where(born, jnp.arange(capacity), -1).astype(ID_DTYPE)
    zeros_i = jnp.zeros((capacity,), dtype=COUNT_DTYPE)
    zeros_f = jnp.zeros((capacity,), dtype=REAL_DTYPE)
    return BirthEventBatch(
        born=born,
        birth_step=zeros_i,
        child_agent_id=ids,
        parent_agent_id=ids,
        lineage_id=ids,
        generation=zeros_i,
        child_genome_id=ids,
        parent_genome_id=ids,
        birth_position=jnp.zeros((capacity, 2), dtype=INDEX_DTYPE),
        child_initial_energy=zeros_f,
        parent_energy_before=jnp.where(born, 10.0, 0.0).astype(REAL_DTYPE),
        parent_energy_after=jnp.where(born, 7.0, 0.0).astype(REAL_DTYPE),
        mutation_selected_count=zeros_i,
        mutation_effective_count=zeros_i,
        mutation_selected_weight_count=zeros_i,
        mutation_selected_bias_count=zeros_i,
        mutation_sum_abs_delta=zeros_f,
        mutation_max_abs_delta=zeros_f,
    )


@pytest.fixture
def metrics_case() -> dict[str, Any]:
    capacity = 4
    final_population = population(
        [True, True, True, False], energy=[4, 8, 6, 100], age=[2, 4, 6, 99]
    )
    world = WorldState(
        resources=jnp.asarray([[1, 2], [3, 4]], dtype=REAL_DTYPE),
        environment=jnp.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=REAL_DTYPE),
        occupancy=jnp.zeros((2, 2), dtype=COUNT_DTYPE),
    )
    before = population([True] * capacity, energy=[1, 2, 3, 4], age=[5, 7, 9, 11])
    pre = build_death_records(
        population=before,
        dies=jnp.asarray([True, False, False, False]),
        cause=jnp.asarray([1, 0, 0, 0], dtype=CODE_DTYPE),
        step=jnp.asarray(2, dtype=COUNT_DTYPE),
    )
    post = build_death_records(
        population=before,
        dies=jnp.asarray([False, True, False, False]),
        cause=jnp.asarray([0, 2, 0, 0], dtype=CODE_DTYPE),
        step=jnp.asarray(2, dtype=COUNT_DTYPE),
    )
    feeding = FeedingResolutionResult(
        population=final_population,
        world=world,
        actions_after_feeding=jnp.zeros(capacity, dtype=CODE_DTYPE),
        feeding_codes=jnp.asarray([1, 2, 0, 3], dtype=CODE_DTYPE),
        resource_demand=jnp.zeros(capacity, dtype=REAL_DTYPE),
        resource_consumed=jnp.asarray([1, 2, 0, 0], dtype=REAL_DTYPE),
        energy_gained=jnp.asarray([2, 4, 0, 0], dtype=REAL_DTYPE),
        contested_resource_cell_count=jnp.asarray(0, dtype=COUNT_DTYPE),
        resource_limited_cell_count=jnp.asarray(0, dtype=COUNT_DTYPE),
        invalid_feeding_input_count=jnp.asarray(0, dtype=COUNT_DTYPE),
    )
    metabolism = PreActionMetabolismResult(
        population=final_population,
        basal_cost_applied=jnp.ones(capacity, dtype=REAL_DTYPE),
        age_incremented=jnp.zeros(capacity, dtype=jnp.bool_),
        invalid_active_energy_count=jnp.asarray(0, dtype=COUNT_DTYPE),
        invalid_active_age_count=jnp.asarray(0, dtype=COUNT_DTYPE),
    )
    action_costs = ActionEnergyCostResult(
        population=final_population,
        movement_cost_applied=jnp.asarray([1, 0, 1, 0], dtype=REAL_DTYPE),
        feeding_cost_applied=jnp.asarray([0, 2, 0, 0], dtype=REAL_DTYPE),
        action_cost_applied=jnp.asarray([1, 2, 1, 0], dtype=REAL_DTYPE),
        invalid_action_cost_input_count=jnp.asarray(0, dtype=COUNT_DTYPE),
    )
    movement = MovementResolutionResult(
        population=final_population,
        world=world,
        actions_after_movement=jnp.zeros(capacity, dtype=CODE_DTYPE),
        movement_codes=jnp.asarray([1, 0, 1, 2], dtype=CODE_DTYPE),
        contested_destination_count=jnp.asarray(0, dtype=COUNT_DTYPE),
        unresolved_priority_collision_destination_count=jnp.asarray(0, dtype=COUNT_DTYPE),
        invalid_movement_input_count=jnp.asarray(0, dtype=COUNT_DTYPE),
        invalid_alive_position_count_after=jnp.asarray(0, dtype=COUNT_DTYPE),
    )
    return {
        "population": final_population,
        "genomes": create_empty_genome_batch(capacity),
        "world": world,
        "births": births(capacity, (2, 3)),
        "pre": pre,
        "post": post,
        "feeding": feeding,
        "metabolism": metabolism,
        "action_costs": action_costs,
        "movement": movement,
    }
