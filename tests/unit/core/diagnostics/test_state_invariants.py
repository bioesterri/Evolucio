# ruff: noqa: ANN001, ANN201, ANN202

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from evolucio.core.diagnostics import (
    StateInvariantCode,
    check_final_state_invariants,
    check_structural_invariants,
)
from evolucio.core.dtypes import ID_DTYPE, REAL_DTYPE


def check(state):
    return check_structural_invariants(
        state=state,
        resource_capacity=jnp.asarray(2, dtype=REAL_DTYPE),
        width=3,
        height=2,
    )


def test_valid_state_report_contract(valid_state) -> None:
    report = check(valid_state)
    assert bool(report.ok)
    assert report.violation_counts.shape == (19,)
    assert report.failed.shape == (19,)
    assert report.total_violation_count.shape == ()
    assert report.violation_counts.dtype == jnp.int32
    assert report.failed.dtype == jnp.bool_


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (lambda s: s.step, jnp.asarray(-1, dtype=jnp.int32), StateInvariantCode.STEP_NEGATIVE),
        (
            lambda s: s.population.agent_id,
            jnp.asarray([-1, 1, -1], dtype=ID_DTYPE),
            StateInvariantCode.ACTIVE_ID_INVALID,
        ),
        (
            lambda s: s.population.agent_id,
            jnp.asarray([0, 0, -1], dtype=ID_DTYPE),
            StateInvariantCode.ACTIVE_AGENT_ID_DUPLICATE,
        ),
        (
            lambda s: s.population.genome_id,
            jnp.asarray([0, 0, -1], dtype=ID_DTYPE),
            StateInvariantCode.ACTIVE_GENOME_ID_DUPLICATE,
        ),
        (
            lambda s: s.population.parent_id,
            jnp.asarray([4, -1, -1], dtype=ID_DTYPE),
            StateInvariantCode.ACTIVE_GENEALOGY_INVALID,
        ),
        (
            lambda s: s.population.position,
            jnp.asarray([[3, 0], [1, 0], [-1, -1]], dtype=jnp.int32),
            StateInvariantCode.ACTIVE_POSITION_INVALID,
        ),
        (
            lambda s: s.population.energy,
            jnp.asarray([jnp.nan, 5, 0], dtype=REAL_DTYPE),
            StateInvariantCode.ACTIVE_ENERGY_NONFINITE,
        ),
        (
            lambda s: s.population.age,
            jnp.asarray([-1, 0, 0], dtype=jnp.int32),
            StateInvariantCode.ACTIVE_AGE_INVALID,
        ),
        (
            lambda s: s.population.energy,
            jnp.asarray([5, 5, 1], dtype=REAL_DTYPE),
            StateInvariantCode.INACTIVE_POPULATION_NONCANONICAL,
        ),
        (
            lambda s: s.world.occupancy,
            jnp.asarray([[-1, 1, 0], [0, 0, 0]], dtype=jnp.int32),
            StateInvariantCode.OCCUPANCY_NEGATIVE,
        ),
        (
            lambda s: s.world.occupancy,
            jnp.zeros((2, 3), dtype=jnp.int32),
            StateInvariantCode.OCCUPANCY_MISMATCH,
        ),
        (
            lambda s: s.world.resources,
            jnp.asarray([[jnp.nan, 1, 1], [1, 1, 1]], dtype=REAL_DTYPE),
            StateInvariantCode.RESOURCE_NONFINITE,
        ),
        (
            lambda s: s.world.resources,
            jnp.asarray([[-1, 1, 1], [1, 1, 1]], dtype=REAL_DTYPE),
            StateInvariantCode.RESOURCE_OUT_OF_RANGE,
        ),
        (
            lambda s: s.world.resources,
            jnp.asarray([[3, 1, 1], [1, 1, 1]], dtype=REAL_DTYPE),
            StateInvariantCode.RESOURCE_OUT_OF_RANGE,
        ),
        (
            lambda s: s.world.environment,
            jnp.asarray([[jnp.inf, 0, 0], [0, 0, 0]], dtype=REAL_DTYPE),
            StateInvariantCode.ENVIRONMENT_NONFINITE,
        ),
        (
            lambda s: s.ids.next_agent_id,
            jnp.asarray(1, dtype=ID_DTYPE),
            StateInvariantCode.ID_COUNTER_INVALID,
        ),
    ],
)
def test_controlled_single_corruptions(valid_state, path, value, code) -> None:
    corrupted = eqx.tree_at(path, valid_state, value)
    report = check(corrupted)
    assert int(report.violation_counts[code]) > 0


def test_genome_corruptions_are_distinguished(valid_state) -> None:
    active = eqx.tree_at(
        lambda s: s.genomes.layer1.weight,
        valid_state,
        valid_state.genomes.layer1.weight.at[0, 0, 0].set(jnp.nan),
    )
    inactive = eqx.tree_at(
        lambda s: s.genomes.layer1.weight,
        valid_state,
        valid_state.genomes.layer1.weight.at[2, 0, 0].set(1),
    )
    assert int(check(active).violation_counts[StateInvariantCode.ACTIVE_GENOME_NONFINITE]) == 1
    assert int(check(inactive).violation_counts[StateInvariantCode.INACTIVE_GENOME_NONZERO]) == 1


@pytest.mark.parametrize("birth_step", [-1, 1])
def test_active_birth_step_must_be_within_state_history(valid_state, birth_step) -> None:
    corrupted = eqx.tree_at(
        lambda s: s.population.birth_step,
        valid_state,
        valid_state.population.birth_step.at[0].set(birth_step),
    )
    assert int(check(corrupted).violation_counts[StateInvariantCode.ACTIVE_AGE_INVALID]) == 1


def test_descendant_parent_id_must_precede_child_and_counter(valid_state) -> None:
    corrupted = eqx.tree_at(
        lambda s: (s.population.generation, s.population.parent_id),
        valid_state,
        (
            valid_state.population.generation.at[0].set(1),
            valid_state.population.parent_id.at[0].set(999),
        ),
    )
    assert int(check(corrupted).violation_counts[StateInvariantCode.ACTIVE_GENEALOGY_INVALID]) == 1


def test_active_founder_lineages_must_be_unique(valid_state) -> None:
    corrupted = eqx.tree_at(
        lambda s: s.population.lineage_id,
        valid_state,
        valid_state.population.lineage_id.at[1].set(0),
    )
    assert int(check(corrupted).violation_counts[StateInvariantCode.ACTIVE_GENEALOGY_INVALID]) == 1


def test_final_viability_is_separate(valid_state) -> None:
    threshold = eqx.tree_at(
        lambda s: s.population.energy,
        valid_state,
        valid_state.population.energy.at[0].set(1),
    )
    age = eqx.tree_at(
        lambda s: s.population.age,
        valid_state,
        valid_state.population.age.at[0].set(10),
    )
    arguments = dict(
        resource_capacity=jnp.asarray(2),
        death_energy_threshold=jnp.asarray(1),
        maximum_age=jnp.asarray(10),
        width=3,
        height=2,
    )
    assert (
        int(
            check_final_state_invariants(state=threshold, **arguments).violation_counts[
                StateInvariantCode.FINAL_ENERGY_NOT_VIABLE
            ]
        )
        == 1
    )
    assert (
        int(
            check_final_state_invariants(state=age, **arguments).violation_counts[
                StateInvariantCode.FINAL_AGE_NOT_VIABLE
            ]
        )
        == 1
    )
    assert bool(check(threshold).ok)


def test_checker_does_not_repair_and_eager_matches_jit(valid_state) -> None:
    corrupted = eqx.tree_at(
        lambda s: s.population.energy,
        valid_state,
        valid_state.population.energy.at[0].set(jnp.inf),
    )
    before = jax.tree.map(lambda value: value.copy(), corrupted)
    eager = check(corrupted)
    compiled = jax.jit(check)(corrupted)
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.array_equal(left, right, equal_nan=True), before, corrupted
        )
    )
    assert jnp.array_equal(eager.violation_counts, compiled.violation_counts)


def test_slot_permutation_preserves_aggregate(valid_state) -> None:
    permutation = jnp.asarray([1, 0, 2])
    population = jax.tree.map(lambda leaf: leaf[permutation], valid_state.population)
    genomes = jax.tree.map(lambda leaf: leaf[permutation], valid_state.genomes)
    permuted = eqx.tree_at(lambda s: (s.population, s.genomes), valid_state, (population, genomes))
    assert jnp.array_equal(check(valid_state).violation_counts, check(permuted).violation_counts)


def test_checker_can_run_inside_scan(valid_state) -> None:
    def body(state, _):
        report = check(state)
        return state, report.total_violation_count

    _, totals = jax.lax.scan(body, valid_state, xs=None, length=2)
    assert jnp.array_equal(totals, jnp.zeros(2, dtype=jnp.int32))
