# ruff: noqa: ANN001, ANN201, ANN202

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.evolution import ReproductionResolutionCode, resolve_asexual_reproduction
from evolucio.core.ids import MAX_NEXT_ID, IdCounters


def test_success_initializes_complete_child_and_debits_total_cost_once(reproduction_case):
    before = reproduction_case
    result = resolve_asexual_reproduction(**before)
    child = int(jnp.flatnonzero(result.newborn_mask, size=1)[0])

    assert int(result.birth_count) == 1
    assert int(result.parent_slots[child]) == 0
    assert int(result.population.agent_id[child]) == 20
    assert int(result.population.genome_id[child]) == 30
    assert int(result.population.parent_id[child]) == 7
    assert int(result.population.lineage_id[child]) == 4
    assert int(result.population.generation[child]) == 3
    assert int(result.population.age[child]) == 0
    assert int(result.population.birth_step[child]) == 12
    assert float(result.population.energy[child]) == 3
    assert float(result.population.energy[0]) == 6
    assert result.ids.next_lineage_id == before["ids"].next_lineage_id
    assert jnp.sum(result.world.occupancy) == jnp.sum(result.population.alive)
    assert jnp.array_equal(result.world.resources, before["world"].resources)
    assert jnp.array_equal(result.world.environment, before["world"].environment)
    assert all(
        jnp.array_equal(after[child], parent[0])
        for after, parent in zip(
            jax.tree.leaves(result.genomes),
            jax.tree.leaves(before["genomes"]),
            strict=True,
        )
    )


def test_no_position_is_atomic(reproduction_case):
    case = dict(reproduction_case)
    case["world"] = eqx.tree_at(
        lambda value: value.occupancy, case["world"], jnp.ones((3, 3), dtype=jnp.int32)
    )
    result = resolve_asexual_reproduction(**case)
    assert int(result.reproduction_codes[0]) == ReproductionResolutionCode.NO_BIRTH_POSITION
    assert int(result.birth_count) == 0
    assert jax.tree.all(jax.tree.map(jnp.array_equal, result.population, case["population"]))
    assert jax.tree.all(jax.tree.map(jnp.array_equal, result.genomes, case["genomes"]))
    assert jax.tree.all(jax.tree.map(jnp.array_equal, result.ids, case["ids"]))
    assert jnp.array_equal(result.world.occupancy, case["world"].occupancy)


def test_id_overflow_rolls_back_everything(reproduction_case):
    case = dict(reproduction_case)
    case["ids"] = IdCounters(
        next_agent_id=jnp.asarray(MAX_NEXT_ID, dtype=jnp.int32),
        next_genome_id=jnp.asarray(MAX_NEXT_ID, dtype=jnp.int32),
        next_lineage_id=case["ids"].next_lineage_id,
    )
    result = resolve_asexual_reproduction(**case)
    assert bool(result.id_overflow)
    assert int(result.reproduction_codes[0]) == ReproductionResolutionCode.ID_OVERFLOW
    assert int(result.birth_count) == 0
    assert jax.tree.all(jax.tree.map(jnp.array_equal, result.population, case["population"]))
    assert jax.tree.all(jax.tree.map(jnp.array_equal, result.ids, case["ids"]))


def test_eager_jit_and_scan_are_equivalent(reproduction_case):
    eager = resolve_asexual_reproduction(**reproduction_case)
    compiled = eqx.filter_jit(resolve_asexual_reproduction)(**reproduction_case)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, eager, compiled))

    def body(carry, unused):
        result = resolve_asexual_reproduction(**{**reproduction_case, "population": carry})
        return result.population, result.birth_count

    _, counts = jax.lax.scan(body, reproduction_case["population"], xs=None, length=1)
    assert int(counts[0]) == 1
