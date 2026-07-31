import jax
import jax.numpy as jnp

from evolucio.core.actions import FeedingResolutionCode
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
