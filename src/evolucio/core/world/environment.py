"""Deterministic resolution and materialisation of global environmental phases."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.dtypes import INDEX_DTYPE, REAL_DTYPE
from evolucio.core.types import Array

if TYPE_CHECKING:
    from evolucio.config.compile import WorldCoreConfig

NO_ACTIVE_PHASE = -1
BASELINE_REGENERATION_MULTIPLIER = 1.0


class EnvironmentControl(eqx.Module):
    """Transient scalar controls resolved for one simulation step."""

    active_phase_index: Array
    regeneration_multiplier: Array
    environment_value: Array


def resolve_environment_control(step: Array, config: WorldCoreConfig) -> EnvironmentControl:
    """Resolve the unique active half-open phase, or the basal controls."""
    calendar = config.environment_calendar
    baseline_index = jnp.asarray(NO_ACTIVE_PHASE, dtype=INDEX_DTYPE)
    baseline_multiplier = jnp.asarray(BASELINE_REGENERATION_MULTIPLIER, dtype=REAL_DTYPE)
    if calendar.phase_count == 0:
        return EnvironmentControl(
            baseline_index, baseline_multiplier, config.environment_initial_value
        )

    active = (calendar.start_steps <= step) & (step < calendar.end_steps)
    has_active = jnp.any(active)
    selected = jnp.argmax(active).astype(INDEX_DTYPE)
    index = jnp.where(has_active, selected, baseline_index)
    multiplier = jnp.where(
        has_active, calendar.regeneration_multipliers[selected], baseline_multiplier
    )
    environment_value = jnp.where(
        has_active, calendar.environment_values[selected], config.environment_initial_value
    )
    return EnvironmentControl(index, multiplier, environment_value)


def update_environment_layer(environment: Array, environment_value: Array) -> Array:
    """Fill the fixed-shape global environmental layer with one scalar value."""
    return jnp.full_like(environment, environment_value, dtype=REAL_DTYPE)
