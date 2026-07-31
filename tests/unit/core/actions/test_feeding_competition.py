import jax
import jax.numpy as jnp
import pytest

from evolucio.core.actions import FeedingResolutionCode, resolve_feeding
from evolucio.core.codes import ActionCode

from .test_feeding import feed, states


def test_equal_competitors_share_resource_and_balances_hold() -> None:
    population, world = states([2, 2], [[2]], [[0, 0], [0, 0]])
    result = feed(population, world, [ActionCode.EAT, ActionCode.EAT])
    assert result.resource_consumed.tolist() == [1, 1]
    assert result.feeding_codes.tolist() == [
        FeedingResolutionCode.FED_PARTIAL,
        FeedingResolutionCode.FED_PARTIAL,
    ]
    assert result.contested_resource_cell_count.item() == 1
    assert result.resource_limited_cell_count.item() == 1
    assert jnp.isclose(
        world.resources.sum(), result.world.resources.sum() + result.resource_consumed.sum()
    )
    assert jnp.isclose(result.energy_gained.sum(), result.resource_consumed.sum() * 2)
    assert jnp.all(result.population.energy <= 10)


def test_different_demands_receive_same_fraction_across_independent_cells() -> None:
    population, world = states([4, 8, 2], [[2, 3]], [[0, 0], [0, 0], [1, 0]])
    result = feed(population, world, [ActionCode.EAT] * 3)
    fractions = result.resource_consumed[:2] / result.resource_demand[:2]
    assert jnp.isclose(fractions[0], fractions[1])
    assert result.resource_consumed[2] == 3
    assert result.resource_limited_cell_count.item() == 1


def test_slot_permutation_is_equivariant() -> None:
    population, world = states([4, 8], [[2]], [[0, 0], [0, 0]])
    first = feed(population, world, [ActionCode.EAT] * 2)
    swapped = jnp.asarray([1, 0])
    permuted = jax.tree.map(lambda value: value[swapped], population)
    second = feed(permuted, world, [ActionCode.EAT] * 2)
    assert jnp.allclose(first.resource_consumed[swapped], second.resource_consumed)


def test_float32_rounding_never_leaves_negative_resource() -> None:
    population, world = states([1.452536, 9.91370277], [[1.5276768]], [[0, 0], [0, 0]])
    result = resolve_feeding(
        population=population,
        world=world,
        actions_after_movement=jnp.asarray([ActionCode.EAT, ActionCode.EAT]),
        maximum_energy=jnp.asarray(10.0, dtype=jnp.float32),
        energy_gain_per_resource=jnp.asarray(1.0, dtype=jnp.float32),
        feeding_max_resource_intake=jnp.asarray(10.0, dtype=jnp.float32),
        width=1,
        height=1,
    )

    assert result.resource_demand.tolist() == pytest.approx([8.547464, 0.08629723], abs=5e-7)
    naive_consumed = result.resource_demand * (
        world.resources.item() / result.resource_demand.sum()
    )
    assert naive_consumed.sum() > world.resources.item()
    assert result.world.resources.item() >= 0
    assert world.resources.sum() == pytest.approx(
        float(result.world.resources.sum() + result.resource_consumed.sum()), abs=2e-7
    )
