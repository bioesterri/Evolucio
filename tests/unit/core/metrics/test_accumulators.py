# ruff: noqa: ANN001, ANN202

import dataclasses

import jax
import jax.numpy as jnp

from evolucio.core.metrics import (
    create_empty_metrics_accumulator,
    create_step_event_buffer,
    update_metrics_accumulator,
)

from .test_step_metrics import compute


def test_empty_then_two_steps_accumulate_and_scan(metrics_case) -> None:
    metrics = compute(metrics_case)
    events = create_step_event_buffer(
        births=metrics_case["births"],
        pre_action_deaths=metrics_case["pre"],
        post_action_deaths=metrics_case["post"],
    )
    empty = create_empty_metrics_accumulator()
    accumulated = update_metrics_accumulator(
        update_metrics_accumulator(empty, metrics, events), metrics, events
    )
    assert accumulated.steps_accumulated.item() == 2
    assert accumulated.births_total.item() == 4
    assert accumulated.birth_rejected_capacity_total.item() == 6
    assert accumulated.deaths_total.item() == 4
    assert accumulated.death_age_sum.item() == 24
    assert accumulated.death_energy_removed_total.item() == 6

    def body(carry, _):
        return update_metrics_accumulator(carry, metrics, events), metrics.alive_count

    scanned, output = jax.lax.scan(body, empty, xs=None, length=2)
    assert scanned.births_total.item() == accumulated.births_total.item()
    assert output.shape == (2,)


def test_counter_overflow_saturates_without_wraparound(metrics_case) -> None:
    metrics = compute(metrics_case)
    events = create_step_event_buffer(
        births=metrics_case["births"],
        pre_action_deaths=metrics_case["pre"],
        post_action_deaths=metrics_case["post"],
    )
    maximum = jnp.iinfo(jnp.int32).max
    accumulator = dataclasses.replace(
        create_empty_metrics_accumulator(), births_total=jnp.asarray(maximum, dtype=jnp.int32)
    )
    updated = update_metrics_accumulator(accumulator, metrics, events)
    assert updated.births_total.item() == maximum
    assert updated.count_overflow.item()
