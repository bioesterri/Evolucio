# ruff: noqa: ANN001, ANN201

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.codes import ActionCode
from evolucio.core.evolution import ReproductionResolutionCode, resolve_asexual_reproduction
from evolucio.core.spatial import rebuild_world_occupancy


def test_no_free_slot_is_atomic(reproduction_case):
    case = dict(reproduction_case)
    population = eqx.tree_at(
        lambda value: (
            value.alive,
            value.agent_id,
            value.lineage_id,
            value.genome_id,
            value.position,
        ),
        case["population"],
        (
            jnp.ones(3, dtype=bool),
            jnp.asarray([7, 8, 9], dtype=jnp.int32),
            jnp.asarray([4, 5, 6], dtype=jnp.int32),
            jnp.asarray([9, 10, 11], dtype=jnp.int32),
            jnp.asarray([[1, 1], [0, 0], [2, 2]], dtype=jnp.int32),
        ),
    )
    case["population"] = population
    case["world"] = rebuild_world_occupancy(case["world"], population, width=3, height=3).world
    result = resolve_asexual_reproduction(**case)
    assert int(result.reproduction_codes[0]) == ReproductionResolutionCode.NO_FREE_POPULATION_SLOT
    assert int(result.birth_count) == 0
    assert jax.tree.all(jax.tree.map(jnp.array_equal, result.population, population))
    assert jax.tree.all(jax.tree.map(jnp.array_equal, result.ids, case["ids"]))
    assert jnp.array_equal(result.world.occupancy, case["world"].occupancy)


def test_same_cell_conflict_has_one_reproducible_identity_winner(reproduction_case):
    case = dict(reproduction_case)
    population = eqx.tree_at(
        lambda value: (
            value.alive,
            value.agent_id,
            value.lineage_id,
            value.genome_id,
            value.position,
            value.energy,
        ),
        case["population"],
        (
            jnp.asarray([True, True, False]),
            jnp.asarray([7, 8, -1], dtype=jnp.int32),
            jnp.asarray([4, 5, -1], dtype=jnp.int32),
            jnp.asarray([9, 10, -1], dtype=jnp.int32),
            jnp.asarray([[0, 1], [2, 1], [-1, -1]], dtype=jnp.int32),
            jnp.asarray([10, 10, 0], dtype=jnp.float32),
        ),
    )
    case["population"] = population
    case["world"] = eqx.tree_at(
        lambda value: value.occupancy,
        case["world"],
        jnp.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.int32),
    )
    case["reproduction_gate"] = eqx.tree_at(
        lambda value: value.eligible,
        case["reproduction_gate"],
        jnp.asarray([True, True, False]),
    )
    case["actions_after_viability"] = jnp.asarray(
        [ActionCode.REPRODUCE, ActionCode.REPRODUCE, ActionCode.STAY], dtype=jnp.int32
    )
    first = resolve_asexual_reproduction(**case)
    second = resolve_asexual_reproduction(**case)
    codes = first.reproduction_codes[:2]
    assert int(first.birth_count) == 1
    assert jnp.sum(codes == ReproductionResolutionCode.BIRTH_SUCCEEDED) == 1
    assert jnp.sum(codes == ReproductionResolutionCode.BIRTH_POSITION_CONFLICT) == 1
    assert jnp.array_equal(first.reproduction_codes, second.reproduction_codes)
    assert jnp.array_equal(first.birth_positions, second.birth_positions)
