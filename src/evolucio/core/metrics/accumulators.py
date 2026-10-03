"""Incremental fixed-shape summaries of step metrics."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.codes import DEATH_CAUSE_COUNT
from evolucio.core.dtypes import COUNT_DTYPE, MASK_DTYPE, REAL_DTYPE
from evolucio.core.types import Array

from .events import StepEventBuffer
from .step import StepMetrics


class MetricsAccumulator(eqx.Module):
    """Compact execution totals suitable for use as a ``lax.scan`` carry."""

    steps_accumulated: Array
    births_total: Array
    birth_rejected_capacity_total: Array
    deaths_total: Array
    deaths_by_cause_total: Array
    resource_consumed_total: Array
    feeding_energy_gained_total: Array
    basal_energy_cost_total: Array
    movement_energy_cost_total: Array
    feeding_energy_cost_total: Array
    reproduction_energy_cost_total: Array
    death_energy_removed_total: Array
    movement_success_total: Array
    feeding_success_total: Array
    reproduction_success_total: Array
    death_age_sum: Array
    alive_count_sum: Array
    energy_alive_sum: Array
    count_overflow: Array


def create_empty_metrics_accumulator() -> MetricsAccumulator:
    """Create the canonical zero-valued accumulator."""
    count = jnp.asarray(0, dtype=COUNT_DTYPE)
    real = jnp.asarray(0, dtype=REAL_DTYPE)
    return MetricsAccumulator(
        steps_accumulated=count,
        births_total=count,
        birth_rejected_capacity_total=count,
        deaths_total=count,
        deaths_by_cause_total=jnp.zeros((DEATH_CAUSE_COUNT,), dtype=COUNT_DTYPE),
        resource_consumed_total=real,
        feeding_energy_gained_total=real,
        basal_energy_cost_total=real,
        movement_energy_cost_total=real,
        feeding_energy_cost_total=real,
        reproduction_energy_cost_total=real,
        death_energy_removed_total=real,
        movement_success_total=count,
        feeding_success_total=count,
        reproduction_success_total=count,
        death_age_sum=real,
        alive_count_sum=count,
        energy_alive_sum=real,
        count_overflow=jnp.asarray(False, dtype=MASK_DTYPE),
    )


def _saturating_add(left: Array, right: Array) -> tuple[Array, Array]:
    maximum = jnp.asarray(jnp.iinfo(COUNT_DTYPE).max, dtype=COUNT_DTYPE)
    overflow = left > maximum - right
    return jnp.where(overflow, maximum, left + right).astype(COUNT_DTYPE), overflow


def update_metrics_accumulator(
    accumulator: MetricsAccumulator, metrics: StepMetrics, events: StepEventBuffer
) -> MetricsAccumulator:
    """Functionally add one step, saturating every int32 counter on overflow."""
    count_pairs = (
        (accumulator.steps_accumulated, jnp.asarray(1, dtype=COUNT_DTYPE)),
        (accumulator.births_total, metrics.birth_count),
        (accumulator.birth_rejected_capacity_total, metrics.birth_rejected_capacity),
        (accumulator.deaths_total, metrics.death_count),
        (accumulator.deaths_by_cause_total, metrics.deaths_by_cause),
        (accumulator.movement_success_total, metrics.movement_success_count),
        (accumulator.feeding_success_total, metrics.feeding_success_count),
        (accumulator.reproduction_success_total, metrics.reproduction_success_count),
        (accumulator.alive_count_sum, metrics.alive_count),
    )
    sums = tuple(_saturating_add(left, right) for left, right in count_pairs)
    overflow = accumulator.count_overflow | events.deaths.overflow
    for _, detected in sums:
        overflow = overflow | jnp.any(detected)
    death_age = jnp.sum(
        jnp.where(events.deaths.records.died, events.deaths.records.age, 0), dtype=REAL_DTYPE
    )
    return MetricsAccumulator(
        steps_accumulated=sums[0][0],
        births_total=sums[1][0],
        birth_rejected_capacity_total=sums[2][0],
        deaths_total=sums[3][0],
        deaths_by_cause_total=sums[4][0],
        resource_consumed_total=accumulator.resource_consumed_total + metrics.resource_consumed,
        feeding_energy_gained_total=(
            accumulator.feeding_energy_gained_total + metrics.feeding_energy_gained
        ),
        basal_energy_cost_total=accumulator.basal_energy_cost_total + metrics.basal_energy_cost,
        movement_energy_cost_total=(
            accumulator.movement_energy_cost_total + metrics.movement_energy_cost
        ),
        feeding_energy_cost_total=(
            accumulator.feeding_energy_cost_total + metrics.feeding_energy_cost
        ),
        reproduction_energy_cost_total=(
            accumulator.reproduction_energy_cost_total + metrics.reproduction_energy_cost
        ),
        death_energy_removed_total=(
            accumulator.death_energy_removed_total + metrics.death_energy_removed
        ),
        movement_success_total=sums[5][0],
        feeding_success_total=sums[6][0],
        reproduction_success_total=sums[7][0],
        death_age_sum=accumulator.death_age_sum + death_age,
        alive_count_sum=sums[8][0],
        energy_alive_sum=accumulator.energy_alive_sum + metrics.total_energy_alive,
        count_overflow=overflow.astype(MASK_DTYPE),
    )
