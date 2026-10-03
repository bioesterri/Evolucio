from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from evolucio.core.codes import DEATH_CAUSE_COUNT
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, ID_DTYPE, INDEX_DTYPE, REAL_DTYPE
from evolucio.core.evolution import BirthEventBatch
from evolucio.core.ids import IdCounters
from evolucio.core.metrics import (
    StepEventBuffer,
    StepMetrics,
    combine_death_records,
    create_empty_metrics_accumulator,
)
from evolucio.core.policy import create_empty_genome_batch
from evolucio.core.rng import RngState, advance_rng, create_rng_state
from evolucio.core.spatial import rebuild_world_occupancy
from evolucio.core.state import PopulationState, SimulationState, WorldState
from evolucio.core.viability import build_death_records


def make_population(alive: list[bool]) -> PopulationState:
    capacity = len(alive)
    mask = jnp.asarray(alive)
    ids = jnp.where(mask, jnp.arange(capacity), -1).astype(ID_DTYPE)
    zeros = jnp.zeros(capacity, dtype=COUNT_DTYPE)
    return PopulationState(
        alive=mask,
        agent_id=ids,
        parent_id=jnp.full(capacity, -1, dtype=ID_DTYPE),
        lineage_id=ids,
        genome_id=ids,
        generation=zeros,
        position=jnp.where(
            mask[:, None], jnp.stack((jnp.arange(capacity), zeros), axis=1), -1
        ).astype(INDEX_DTYPE),
        energy=jnp.where(mask, 5.0, 0.0).astype(REAL_DTYPE),
        birth_step=zeros,
        age=zeros,
    )


def make_state(alive: list[bool], *, step: int = 0, rng: RngState | None = None) -> SimulationState:
    population = make_population(alive)
    capacity = len(alive)
    world = WorldState(
        resources=jnp.ones((2, capacity), dtype=REAL_DTYPE),
        environment=jnp.zeros((2, capacity), dtype=REAL_DTYPE),
        occupancy=jnp.zeros((2, capacity), dtype=COUNT_DTYPE),
    )
    world = rebuild_world_occupancy(world, population, width=capacity, height=2).world
    active_count = sum(alive)
    return SimulationState(
        step=jnp.asarray(step, dtype=COUNT_DTYPE),
        rng=create_rng_state(7) if rng is None else rng,
        ids=IdCounters(
            next_agent_id=jnp.asarray(active_count, dtype=ID_DTYPE),
            next_genome_id=jnp.asarray(active_count, dtype=ID_DTYPE),
            next_lineage_id=jnp.asarray(active_count, dtype=ID_DTYPE),
        ),
        world=world,
        population=population,
        genomes=create_empty_genome_batch(capacity),
    )


def make_births(capacity: int, slot: int | None = None) -> BirthEventBatch:
    born = jnp.arange(capacity) == (-1 if slot is None else slot)
    zeros_i = jnp.zeros(capacity, dtype=COUNT_DTYPE)
    zeros_f = jnp.zeros(capacity, dtype=REAL_DTYPE)
    ids = jnp.where(born, slot if slot is not None else -1, -1).astype(ID_DTYPE)
    return BirthEventBatch(
        born=born,
        birth_step=zeros_i,
        child_agent_id=ids,
        parent_agent_id=jnp.where(born, 0, -1).astype(ID_DTYPE),
        lineage_id=jnp.where(born, 0, -1).astype(ID_DTYPE),
        generation=jnp.where(born, 1, 0).astype(COUNT_DTYPE),
        child_genome_id=ids,
        parent_genome_id=jnp.where(born, 0, -1).astype(ID_DTYPE),
        birth_position=jnp.where(born[:, None], jnp.asarray([slot or 0, 0]), -1).astype(
            INDEX_DTYPE
        ),
        child_initial_energy=jnp.where(born, 5, 0).astype(REAL_DTYPE),
        parent_energy_before=zeros_f,
        parent_energy_after=zeros_f,
        mutation_selected_count=zeros_i,
        mutation_effective_count=zeros_i,
        mutation_selected_weight_count=zeros_i,
        mutation_selected_bias_count=zeros_i,
        mutation_sum_abs_delta=zeros_f,
        mutation_max_abs_delta=zeros_f,
    )


def make_metrics(alive: int, births: int, deaths: int) -> StepMetrics:
    count = jnp.asarray(0, dtype=COUNT_DTYPE)
    real = jnp.asarray(0, dtype=REAL_DTYPE)
    return StepMetrics(
        step=count,
        alive_count=jnp.asarray(alive, dtype=COUNT_DTYPE),
        birth_count=jnp.asarray(births, dtype=COUNT_DTYPE),
        birth_rejected_capacity=count,
        death_count=jnp.asarray(deaths, dtype=COUNT_DTYPE),
        deaths_by_cause=jnp.zeros(DEATH_CAUSE_COUNT, dtype=COUNT_DTYPE),
        total_energy_alive=real,
        mean_energy_alive=real,
        mean_age_alive=real,
        mean_age_at_death=real,
        total_resources=real,
        resource_consumed=real,
        feeding_energy_gained=real,
        basal_energy_cost=real,
        movement_energy_cost=real,
        feeding_energy_cost=real,
        reproduction_energy_cost=real,
        death_energy_removed=real,
        movement_success_count=count,
        feeding_success_count=count,
        reproduction_success_count=jnp.asarray(births, dtype=COUNT_DTYPE),
        active_lineage_count=count,
        genetic_parameter_variance=real,
        mean_environment=real,
    )


@pytest.fixture
def valid_state() -> SimulationState:
    return make_state([True, True, False])


@pytest.fixture
def transition_case() -> dict[str, Any]:
    before = make_state([True, True, False])
    next_rng, _ = advance_rng(before.rng)
    after = make_state([True, True, True], step=1, rng=next_rng)
    after = jax.tree.map(lambda x: x, after)
    population = after.population
    population = PopulationState(
        alive=population.alive,
        agent_id=population.agent_id,
        parent_id=population.parent_id.at[2].set(0),
        lineage_id=population.lineage_id.at[2].set(0),
        genome_id=population.genome_id,
        generation=population.generation.at[2].set(1),
        position=population.position,
        energy=population.energy,
        birth_step=population.birth_step,
        age=population.age,
    )
    after = SimulationState(
        step=after.step,
        rng=after.rng,
        ids=IdCounters(
            next_agent_id=jnp.asarray(3, dtype=ID_DTYPE),
            next_genome_id=jnp.asarray(3, dtype=ID_DTYPE),
            next_lineage_id=jnp.asarray(2, dtype=ID_DTYPE),
        ),
        world=after.world,
        population=population,
        genomes=after.genomes,
    )
    empty_deaths = build_death_records(
        population=before.population,
        dies=jnp.zeros(3, dtype=jnp.bool_),
        cause=jnp.zeros(3, dtype=CODE_DTYPE),
        step=jnp.asarray(0, dtype=COUNT_DTYPE),
    )
    event_buffer = StepEventBuffer(
        births=make_births(3, 2),
        deaths=combine_death_records(pre_action=empty_deaths, post_action=empty_deaths),
    )
    return {
        "before": before,
        "after": after,
        "events": event_buffer,
        "metrics": make_metrics(3, 1, 0),
        "accumulator": create_empty_metrics_accumulator(),
    }
