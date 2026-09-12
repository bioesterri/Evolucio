# ruff: noqa: ANN001, ANN003, ANN201, ANN202
import inspect

import jax
import jax.numpy as jnp
import pytest

from evolucio.core.codes import ActionCode
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, MASK_DTYPE, REAL_DTYPE
from evolucio.core.state import PopulationState
from evolucio.core.viability import ReproductionGateCode, evaluate_reproduction_gate


def changed(population, slot=0, **changes):
    values = vars(population).copy()
    for field, value in changes.items():
        values[field] = values[field].at[slot].set(jnp.asarray(value, dtype=values[field].dtype))
    return PopulationState(**values)


def gate(
    before,
    after=None,
    actions=None,
    *,
    threshold=5.0,
    age=2,
    cost=4.0,
    offspring_energy=2.0,
    death=1.0,
):
    if after is None:
        after = before
    if actions is None:
        actions = jnp.asarray(
            [ActionCode.REPRODUCE, ActionCode.EAT, ActionCode.STAY], dtype=CODE_DTYPE
        )
    return evaluate_reproduction_gate(
        population_before_viability=before,
        population_after_viability=after,
        actions_before_viability=actions,
        actions_after_viability=actions,
        reproduction_energy_threshold=jnp.asarray(threshold, dtype=REAL_DTYPE),
        minimum_reproduction_age=jnp.asarray(age, dtype=COUNT_DTYPE),
        reproduction_energy_cost=jnp.asarray(cost, dtype=REAL_DTYPE),
        offspring_initial_energy=jnp.asarray(offspring_energy, dtype=REAL_DTYPE),
        death_energy_threshold=jnp.asarray(death, dtype=REAL_DTYPE),
    )


def test_critical_suicidal_boundary_does_not_charge_parent(viability_state):
    population, _, _ = viability_state
    result = gate(population)
    assert not bool(result.eligible[0])
    assert int(result.gate_codes[0]) == ReproductionGateCode.SUICIDAL_PROJECTED_ENERGY
    assert float(result.projected_parent_energy[0]) == -1.0
    assert float(population.energy[0]) == 5.0
    assert int(result.suicidal_block_count) == 1

    above = changed(population, energy=7.1)
    eligible = gate(above)
    assert bool(eligible.eligible[0])
    assert int(eligible.gate_codes[0]) == ReproductionGateCode.ELIGIBLE
    assert float(above.energy[0]) == pytest.approx(7.1)


def test_projection_debits_reproduction_cost_and_offspring_energy(viability_state):
    population, _, _ = viability_state
    population = changed(population, energy=10.0)

    result = gate(
        population,
        threshold=10.0,
        cost=5.0,
        offspring_energy=10.0,
        death=0.0,
    )

    assert not bool(result.eligible[0])
    assert int(result.gate_codes[0]) == ReproductionGateCode.SUICIDAL_PROJECTED_ENERGY
    assert float(result.projected_parent_energy[0]) == -5.0
    assert int(result.requested_count) == 1
    assert int(result.eligible_count) == 0
    assert int(result.suicidal_block_count) == 1
    assert float(population.energy[0]) == 10.0


@pytest.mark.parametrize(
    ("changes", "kwargs", "code"),
    [
        ({"energy": 4.9}, {}, ReproductionGateCode.BELOW_ENERGY_THRESHOLD),
        ({"age": 1}, {}, ReproductionGateCode.BELOW_MINIMUM_AGE),
        ({"energy": 5.0}, {"cost": 4.1}, ReproductionGateCode.SUICIDAL_PROJECTED_ENERGY),
        ({}, {"cost": float("nan")}, ReproductionGateCode.INVALID_REPRODUCTION_INPUT),
    ],
)
def test_blocking_codes(viability_state, changes, kwargs, code):
    population, _, _ = viability_state
    result = gate(changed(population, **changes), **kwargs)
    assert int(result.gate_codes[0]) == code
    assert not bool(result.eligible[0])


def test_dead_request_is_distinct_from_not_requested(viability_state):
    before, _, _ = viability_state
    after = changed(before, alive=False, agent_id=-1, energy=0.0, position=[-1, -1])
    actions_after = jnp.asarray(
        [ActionCode.STAY, ActionCode.EAT, ActionCode.STAY], dtype=CODE_DTYPE
    )
    result = evaluate_reproduction_gate(
        population_before_viability=before,
        population_after_viability=after,
        actions_before_viability=jnp.asarray(
            [ActionCode.REPRODUCE, ActionCode.EAT, ActionCode.STAY], dtype=CODE_DTYPE
        ),
        actions_after_viability=actions_after,
        reproduction_energy_threshold=jnp.asarray(5, dtype=REAL_DTYPE),
        minimum_reproduction_age=jnp.asarray(2, dtype=COUNT_DTYPE),
        reproduction_energy_cost=jnp.asarray(1, dtype=REAL_DTYPE),
        offspring_initial_energy=jnp.asarray(2, dtype=REAL_DTYPE),
        death_energy_threshold=jnp.asarray(0, dtype=REAL_DTYPE),
    )
    assert result.gate_codes.tolist() == [ReproductionGateCode.DEAD_POST_ACTION, 0, 0]
    assert int(result.requested_count) == 1
    assert not bool(jnp.any(result.eligible))


def test_gate_precedence_is_explicit(viability_state):
    population, _, _ = viability_state
    dead = changed(population, alive=False)
    result = gate(population, after=dead, threshold=float("nan"), cost=float("nan"))
    assert int(result.gate_codes[0]) == ReproductionGateCode.DEAD_POST_ACTION

    not_requested = jnp.full((3,), ActionCode.STAY, dtype=CODE_DTYPE)
    result = gate(population, actions=not_requested, threshold=float("nan"))
    assert int(result.gate_codes[0]) == ReproductionGateCode.NOT_REQUESTED


def test_non_reproductive_actions_are_not_requested(viability_state):
    population, _, _ = viability_state
    for action in [ActionCode.STAY, ActionCode.MOVE_NORTH, ActionCode.EAT]:
        actions = jnp.full((3,), action, dtype=CODE_DTYPE)
        result = gate(population, actions=actions)
        assert result.gate_codes.tolist() == [0, 0, 0]
        assert result.projected_parent_energy.tolist() == [0.0, 0.0, 0.0]


def test_shapes_dtypes_jit_scan_permutation_and_no_rng(viability_state):
    population, _, _ = viability_state
    eager = gate(population, cost=1.0, death=0.0)
    compiled = jax.jit(gate)(population, cost=1.0, death=0.0)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, eager, compiled))
    assert (
        eager.eligible.shape
        == eager.gate_codes.shape
        == eager.projected_parent_energy.shape
        == (3,)
    )
    assert eager.eligible.dtype == jnp.dtype(MASK_DTYPE)
    assert eager.gate_codes.dtype == jnp.dtype(CODE_DTYPE)
    assert eager.projected_parent_energy.dtype == jnp.dtype(REAL_DTYPE)
    assert eager.requested_count.shape == eager.eligible_count.shape == ()

    permutation = jnp.asarray([1, 0, 2])
    permuted = jax.tree.map(lambda value: value[permutation], population)
    permuted_result = gate(
        permuted,
        actions=jnp.asarray(
            [ActionCode.EAT, ActionCode.REPRODUCE, ActionCode.STAY], dtype=CODE_DTYPE
        ),
        cost=1.0,
        death=0.0,
    )
    assert jnp.array_equal(permuted_result.eligible, eager.eligible[permutation])

    def body(carry, _):
        result = gate(carry, cost=1.0, death=0.0)
        return carry, result.eligible_count

    _, counts = jax.lax.scan(body, population, xs=None, length=2)
    assert counts.shape == (2,)
    assert "rng" not in inspect.signature(evaluate_reproduction_gate).parameters


def test_enum_values_are_explicit_contiguous_and_unique():
    assert [code.value for code in ReproductionGateCode] == list(range(7))
