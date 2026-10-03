"""Pure quantitative observation of a completed simulation step."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.actions import (
    FeedingResolutionCode,
    FeedingResolutionResult,
    MovementResolutionCode,
    MovementResolutionResult,
)
from evolucio.core.codes import DEATH_CAUSE_COUNT, DeathCauseCode
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, REAL_DTYPE, STEP_DTYPE
from evolucio.core.energy import ActionEnergyCostResult, PreActionMetabolismResult
from evolucio.core.evolution import ReproductionResolutionResult
from evolucio.core.policy import GenomeBatch
from evolucio.core.state import PopulationState, WorldState
from evolucio.core.types import Array

from .events import StepEventBuffer


class StepMetrics(eqx.Module):
    """Final-state gauges and within-step flows for one completed step."""

    step: Array
    alive_count: Array
    birth_count: Array
    birth_rejected_capacity: Array
    death_count: Array
    deaths_by_cause: Array
    total_energy_alive: Array
    mean_energy_alive: Array
    mean_age_alive: Array
    mean_age_at_death: Array
    total_resources: Array
    resource_consumed: Array
    feeding_energy_gained: Array
    basal_energy_cost: Array
    movement_energy_cost: Array
    feeding_energy_cost: Array
    reproduction_energy_cost: Array
    death_energy_removed: Array
    movement_success_count: Array
    feeding_success_count: Array
    reproduction_success_count: Array
    active_lineage_count: Array
    genetic_parameter_variance: Array
    mean_environment: Array


def _safe_mean(total: Array, count: Array) -> Array:
    return jnp.where(count > 0, total / jnp.maximum(count, 1).astype(REAL_DTYPE), 0).astype(
        REAL_DTYPE
    )


def _active_lineage_count(population: PopulationState, alive_count: Array) -> Array:
    sentinel = jnp.asarray(jnp.iinfo(population.lineage_id.dtype).max, population.lineage_id.dtype)
    sorted_lineages = jnp.sort(jnp.where(population.alive, population.lineage_id, sentinel))
    valid_positions = jnp.arange(sorted_lineages.shape[0], dtype=COUNT_DTYPE) < alive_count
    changes = jnp.concatenate(
        (jnp.ones((1,), dtype=jnp.bool_), sorted_lineages[1:] != sorted_lineages[:-1])
    )
    return jnp.sum(valid_positions & changes, dtype=COUNT_DTYPE)


def _genetic_parameter_variance(genomes: GenomeBatch, alive: Array, count: Array) -> Array:
    parameters = jnp.concatenate(
        (
            genomes.layer1.weight.reshape((alive.shape[0], -1)),
            genomes.layer1.bias.reshape((alive.shape[0], -1)),
            genomes.layer2.weight.reshape((alive.shape[0], -1)),
            genomes.layer2.bias.reshape((alive.shape[0], -1)),
        ),
        axis=1,
    )
    mask = alive[:, None].astype(REAL_DTYPE)
    denominator = jnp.maximum(count, 1).astype(REAL_DTYPE)
    means = jnp.sum(parameters * mask, axis=0, dtype=REAL_DTYPE) / denominator
    variances = (
        jnp.sum(jnp.square(parameters - means) * mask, axis=0, dtype=REAL_DTYPE) / denominator
    )
    return jnp.where(count > 1, jnp.mean(variances, dtype=REAL_DTYPE), 0).astype(REAL_DTYPE)


def compute_step_metrics(
    *,
    step: Array,
    population: PopulationState,
    genomes: GenomeBatch,
    world: WorldState,
    events: StepEventBuffer,
    feeding: FeedingResolutionResult,
    metabolism: PreActionMetabolismResult,
    action_costs: ActionEnergyCostResult,
    movement: MovementResolutionResult,
    reproduction: ReproductionResolutionResult,
) -> StepMetrics:
    """Observe final-state gauges and flows without changing simulation inputs."""
    alive_count = jnp.sum(population.alive, dtype=COUNT_DTYPE)
    total_energy = jnp.sum(jnp.where(population.alive, population.energy, 0), dtype=REAL_DTYPE)
    total_age = jnp.sum(jnp.where(population.alive, population.age, 0), dtype=REAL_DTYPE)
    births = events.births.born
    birth_count = jnp.sum(births, dtype=COUNT_DTYPE)
    death_records = events.deaths.records
    death_count = events.deaths.count.astype(COUNT_DTYPE)
    cause_indices = jnp.arange(DEATH_CAUSE_COUNT, dtype=CODE_DTYPE)
    deaths_by_cause = (
        jnp.sum(
            death_records.died[:, None] & (death_records.cause[:, None] == cause_indices[None, :]),
            axis=0,
            dtype=COUNT_DTYPE,
        )
        .at[int(DeathCauseCode.NONE)]
        .set(0)
    )
    death_age_sum = jnp.sum(jnp.where(death_records.died, death_records.age, 0), dtype=REAL_DTYPE)
    feeding_success = (feeding.feeding_codes == FeedingResolutionCode.FED_FULL) | (
        feeding.feeding_codes == FeedingResolutionCode.FED_PARTIAL
    )
    reproduction_cost = jnp.sum(
        jnp.where(
            births,
            events.births.parent_energy_before - events.births.parent_energy_after,
            0,
        ),
        dtype=REAL_DTYPE,
    )
    return StepMetrics(
        step=jnp.asarray(step, dtype=STEP_DTYPE),
        alive_count=alive_count,
        birth_count=birth_count,
        birth_rejected_capacity=reproduction.no_free_slot_count.astype(COUNT_DTYPE),
        death_count=death_count,
        deaths_by_cause=deaths_by_cause,
        total_energy_alive=total_energy,
        mean_energy_alive=_safe_mean(total_energy, alive_count),
        mean_age_alive=_safe_mean(total_age, alive_count),
        mean_age_at_death=_safe_mean(death_age_sum, death_count),
        total_resources=jnp.sum(world.resources, dtype=REAL_DTYPE),
        resource_consumed=jnp.sum(feeding.resource_consumed, dtype=REAL_DTYPE),
        feeding_energy_gained=jnp.sum(feeding.energy_gained, dtype=REAL_DTYPE),
        basal_energy_cost=jnp.sum(metabolism.basal_cost_applied, dtype=REAL_DTYPE),
        movement_energy_cost=jnp.sum(action_costs.movement_cost_applied, dtype=REAL_DTYPE),
        feeding_energy_cost=jnp.sum(action_costs.feeding_cost_applied, dtype=REAL_DTYPE),
        reproduction_energy_cost=reproduction_cost,
        death_energy_removed=jnp.sum(
            jnp.where(death_records.died, death_records.energy, 0), dtype=REAL_DTYPE
        ),
        movement_success_count=jnp.sum(
            movement.movement_codes == MovementResolutionCode.MOVED, dtype=COUNT_DTYPE
        ),
        feeding_success_count=jnp.sum(feeding_success, dtype=COUNT_DTYPE),
        reproduction_success_count=birth_count,
        active_lineage_count=_active_lineage_count(population, alive_count),
        genetic_parameter_variance=_genetic_parameter_variance(
            genomes, population.alive, alive_count
        ),
        mean_environment=jnp.mean(world.environment, dtype=REAL_DTYPE),
    )
