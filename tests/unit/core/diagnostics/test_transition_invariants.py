# ruff: noqa: ANN001, ANN201, ANN202

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from evolucio.core.diagnostics import (
    TransitionInvariantCode,
    check_step_transition_invariants,
)
from evolucio.core.dtypes import COUNT_DTYPE, ID_DTYPE


def check(case):
    return check_step_transition_invariants(**case)


def changed(case, key, path, value):
    result = dict(case)
    result[key] = eqx.tree_at(path, case[key], value)
    return result


def test_valid_transition_balance_and_report(transition_case) -> None:
    report = check(transition_case)
    assert bool(report.ok)
    assert report.violation_counts.shape == (16,)
    assert report.failed.dtype == jnp.bool_
    assert report.total_violation_count.dtype == jnp.int32


@pytest.mark.parametrize(
    ("key", "path", "value", "code"),
    [
        (
            "after",
            lambda s: s.step,
            jnp.asarray(0, dtype=COUNT_DTYPE),
            TransitionInvariantCode.STEP_ADVANCE_MISMATCH,
        ),
        (
            "after",
            lambda s: s.rng.key,
            jax.random.key(99),
            TransitionInvariantCode.RNG_ADVANCE_MISMATCH,
        ),
        (
            "after",
            lambda s: s.ids.next_agent_id,
            jnp.asarray(1, dtype=ID_DTYPE),
            TransitionInvariantCode.ID_COUNTER_REGRESSION,
        ),
        (
            "after",
            lambda s: s.ids.next_agent_id,
            jnp.asarray(4, dtype=ID_DTYPE),
            TransitionInvariantCode.AGENT_ID_ALLOCATION_MISMATCH,
        ),
        (
            "after",
            lambda s: s.ids.next_genome_id,
            jnp.asarray(4, dtype=ID_DTYPE),
            TransitionInvariantCode.GENOME_ID_ALLOCATION_MISMATCH,
        ),
        (
            "after",
            lambda s: s.ids.next_lineage_id,
            jnp.asarray(3, dtype=ID_DTYPE),
            TransitionInvariantCode.LINEAGE_COUNTER_CHANGED,
        ),
        (
            "after",
            lambda s: s.population.alive,
            jnp.asarray([True, True, False]),
            TransitionInvariantCode.POPULATION_BALANCE_MISMATCH,
        ),
        (
            "events",
            lambda e: e.births.born,
            jnp.asarray([False, False, False]),
            TransitionInvariantCode.BIRTH_EVENT_COUNT_MISMATCH,
        ),
        (
            "events",
            lambda e: e.deaths.count,
            jnp.asarray(1, dtype=COUNT_DTYPE),
            TransitionInvariantCode.DEATH_EVENT_COUNT_MISMATCH,
        ),
        (
            "metrics",
            lambda m: m.alive_count,
            jnp.asarray(2, dtype=COUNT_DTYPE),
            TransitionInvariantCode.METRIC_ALIVE_COUNT_MISMATCH,
        ),
        (
            "metrics",
            lambda m: m.birth_count,
            jnp.asarray(0, dtype=COUNT_DTYPE),
            TransitionInvariantCode.METRIC_BIRTH_COUNT_MISMATCH,
        ),
        (
            "metrics",
            lambda m: m.death_count,
            jnp.asarray(1, dtype=COUNT_DTYPE),
            TransitionInvariantCode.METRIC_DEATH_COUNT_MISMATCH,
        ),
        (
            "events",
            lambda e: e.deaths.overflow,
            jnp.asarray(True),
            TransitionInvariantCode.EVENT_BUFFER_OVERFLOW,
        ),
        (
            "accumulator",
            lambda a: a.count_overflow,
            jnp.asarray(True),
            TransitionInvariantCode.METRICS_ACCUMULATOR_OVERFLOW,
        ),
    ],
)
def test_transition_corruptions(transition_case, key, path, value, code) -> None:
    report = check(changed(transition_case, key, path, value))
    assert int(report.violation_counts[code]) > 0


def test_death_causes_and_metric_histogram_are_checked(transition_case) -> None:
    died = jnp.asarray([True, False, False])
    events = eqx.tree_at(
        lambda e: (e.deaths.records.died, e.deaths.count),
        transition_case["events"],
        (died, jnp.asarray(1, dtype=COUNT_DTYPE)),
    )
    case = dict(transition_case, events=events)
    report = check(case)
    assert int(report.violation_counts[TransitionInvariantCode.DEATH_CAUSE_COUNT_MISMATCH]) == 1
    metrics = eqx.tree_at(
        lambda m: m.deaths_by_cause,
        transition_case["metrics"],
        jnp.asarray([0, 1, 0, 0, 0, 0], dtype=COUNT_DTYPE),
    )
    report = check(dict(transition_case, metrics=metrics))
    assert int(report.violation_counts[TransitionInvariantCode.METRIC_DEATH_CAUSE_MISMATCH]) == 1


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (lambda e: e.births.parent_agent_id, jnp.asarray([-1, -1, 999], dtype=ID_DTYPE)),
        (lambda e: e.births.lineage_id, jnp.asarray([-1, -1, 999], dtype=ID_DTYPE)),
        (
            lambda e: e.births.birth_position,
            jnp.asarray([[-1, -1], [-1, -1], [0, 1]], dtype=jnp.int32),
        ),
        (lambda e: e.births.child_initial_energy, jnp.asarray([0.0, 0.0, 99.0])),
    ],
)
def test_birth_event_content_must_match_newborn_slot(transition_case, path, value) -> None:
    report = check(changed(transition_case, "events", path, value))
    assert int(report.violation_counts[TransitionInvariantCode.BIRTH_EVENT_COUNT_MISMATCH]) > 0


def test_death_event_identity_must_identify_a_removed_agent(transition_case) -> None:
    events = eqx.tree_at(
        lambda e: (
            e.deaths.records.died,
            e.deaths.records.agent_id,
            e.deaths.records.cause,
            e.deaths.count,
        ),
        transition_case["events"],
        (
            jnp.asarray([True, False, False]),
            jnp.asarray([999, -1, -1], dtype=ID_DTYPE),
            jnp.asarray([1, 0, 0], dtype=jnp.int8),
            jnp.asarray(1, dtype=COUNT_DTYPE),
        ),
    )
    report = check(dict(transition_case, events=events))
    assert int(report.violation_counts[TransitionInvariantCode.DEATH_EVENT_COUNT_MISMATCH]) > 0


def test_transition_eager_jit_and_scan(transition_case) -> None:
    eager = check(transition_case)
    compiled = jax.jit(check_step_transition_invariants)(**transition_case)
    assert jnp.array_equal(eager.violation_counts, compiled.violation_counts)

    def body(carry, _):
        report = check_step_transition_invariants(**carry)
        return carry, report.ok

    _, values = jax.lax.scan(body, transition_case, xs=None, length=2)
    assert jnp.all(values)
