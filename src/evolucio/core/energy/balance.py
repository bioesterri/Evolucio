"""Vectorized energy-balance reporting."""

# pyright: reportUnknownMemberType=false

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.dtypes import REAL_DTYPE
from evolucio.core.types import Array


class EnergyBalanceReport(eqx.Module):
    """Per-slot residuals and scalar totals for a tracked cohort."""

    expected_energy_after: Array
    residual: Array
    total_energy_before: Array
    total_energy_after: Array
    total_basal_cost: Array
    total_feeding_gain: Array
    total_movement_cost: Array
    total_feeding_cost: Array
    max_abs_residual: Array


def build_energy_balance_report(
    *,
    energy_before: Array,
    energy_after: Array,
    basal_cost_applied: Array,
    feeding_energy_gained: Array,
    movement_cost_applied: Array,
    feeding_cost_applied: Array,
    tracked_mask: Array,
) -> EnergyBalanceReport:
    """Build an exact, tolerance-free report over the selected fixed slots."""
    expected = (
        energy_before
        - basal_cost_applied
        + feeding_energy_gained
        - movement_cost_applied
        - feeding_cost_applied
    ).astype(REAL_DTYPE)
    residual = (energy_after - expected).astype(REAL_DTYPE)
    zero = jnp.asarray(0, dtype=REAL_DTYPE)

    def total(values: Array) -> Array:
        return jnp.sum(jnp.where(tracked_mask, values, zero), dtype=REAL_DTYPE)

    tracked_residual = jnp.where(tracked_mask, jnp.abs(residual), zero)
    return EnergyBalanceReport(
        expected_energy_after=jnp.where(tracked_mask, expected, zero).astype(REAL_DTYPE),
        residual=jnp.where(tracked_mask, residual, zero).astype(REAL_DTYPE),
        total_energy_before=total(energy_before),
        total_energy_after=total(energy_after),
        total_basal_cost=total(basal_cost_applied),
        total_feeding_gain=total(feeding_energy_gained),
        total_movement_cost=total(movement_cost_applied),
        total_feeding_cost=total(feeding_cost_applied),
        max_abs_residual=jnp.max(tracked_residual, initial=zero),
    )
