# ruff: noqa: ANN001

import jax
import jax.numpy as jnp

from evolucio.core.metrics import DeathPhaseCode, create_step_event_buffer


def test_deaths_from_both_phases_are_compacted_with_fixed_shape(metrics_case) -> None:
    events = create_step_event_buffer(
        births=metrics_case["births"],
        pre_action_deaths=metrics_case["pre"],
        post_action_deaths=metrics_case["post"],
    )

    assert events.deaths.count.item() == 2
    assert events.deaths.records.died.shape == (4,)
    assert events.deaths.records.position.shape == (4, 2)
    assert events.deaths.records.agent_id[:2].tolist() == [0, 1]
    assert events.deaths.phase.tolist() == [
        DeathPhaseCode.PRE_ACTION,
        DeathPhaseCode.POST_ACTION,
        0,
        0,
    ]
    assert not events.deaths.overflow.item()


def test_duplicate_death_is_not_emitted_twice_and_raw_overflow_is_visible(metrics_case) -> None:
    duplicated = create_step_event_buffer(
        births=metrics_case["births"],
        pre_action_deaths=metrics_case["pre"],
        post_action_deaths=metrics_case["pre"],
    )
    assert duplicated.deaths.count.item() == 1

    full_post = jax.tree.map(
        lambda value: jnp.ones_like(value) if value.dtype == jnp.bool_ else value,
        metrics_case["post"],
    )
    overflow = create_step_event_buffer(
        births=metrics_case["births"],
        pre_action_deaths=metrics_case["pre"],
        post_action_deaths=full_post,
    )
    assert overflow.deaths.overflow.item()
