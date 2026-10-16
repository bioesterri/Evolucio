# ruff: noqa: ANN001

import equinox as eqx
import jax.numpy as jnp

from evolucio.core.metrics import create_empty_metrics_accumulator
from evolucio.core.simulation import simulation_step_checked


def test_checked_step_reports_valid_transition(step_case) -> None:
    state, config = step_case()
    checked = simulation_step_checked(
        state=state, accumulator=create_empty_metrics_accumulator(), config=config
    )
    assert bool(checked.initial_invariants.ok)
    assert bool(checked.final_invariants.ok)
    assert bool(checked.transition_invariants.ok)


def test_checked_step_detects_but_does_not_repair_initial_corruption(step_case) -> None:
    state, config = step_case(alive=False)
    corrupted = eqx.tree_at(
        lambda s: s.world.occupancy,
        state,
        state.world.occupancy.at[0, 0].set(jnp.asarray(1, dtype=jnp.int32)),
    )
    checked = simulation_step_checked(
        state=corrupted, accumulator=create_empty_metrics_accumulator(), config=config
    )
    assert not bool(checked.initial_invariants.ok)
    assert corrupted.world.occupancy[0, 0].item() == 1
