# ruff: noqa: ANN001, ANN202

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.codes import ActionCode
from evolucio.core.metrics import create_empty_metrics_accumulator
from evolucio.core.rng import advance_rng
from evolucio.core.simulation import simulation_step


def same_tree(left, right) -> bool:
    return jax.tree.all(jax.tree.map(jnp.array_equal, left, right))


def test_empty_population_advances_rng_and_step_and_supports_jit_scan(step_case) -> None:
    state, config = step_case(alive=False)
    accumulator = create_empty_metrics_accumulator()
    eager = simulation_step(state=state, accumulator=accumulator, config=config)
    compiled = eqx.filter_jit(simulation_step)(state=state, accumulator=accumulator, config=config)
    expected_rng, _ = advance_rng(state.rng)
    assert eager.state.step.item() == 4
    assert eager.metrics.step.item() == 3
    assert same_tree(eager.state.rng, expected_rng)
    assert same_tree(eager, compiled)

    def body(carry, _):
        scan_state, scan_accumulator = carry
        result = simulation_step(state=scan_state, accumulator=scan_accumulator, config=config)
        return (result.state, result.accumulator), (result.metrics, result.events)

    (final_state, final_accumulator), (metrics, _) = jax.lax.scan(
        body, (state, accumulator), xs=None, length=2
    )
    assert final_state.step.item() == 5
    assert final_accumulator.steps_accumulated.item() == 2
    assert metrics.step.tolist() == [3, 4]


def test_preaction_death_cannot_move_feed_or_reproduce(step_case) -> None:
    state, config = step_case(
        action=ActionCode.MOVE_EAST,
        energy=0.05,
        energy_changes={"basal_cost": 0.1},
    )
    result = simulation_step(
        state=state, accumulator=create_empty_metrics_accumulator(), config=config
    )
    assert result.metrics.death_count.item() == 1
    assert result.events.deaths.records.death_step[0].item() == 3
    assert result.metrics.movement_success_count.item() == 0
    assert result.metrics.resource_consumed.item() == 0
    assert result.metrics.birth_count.item() == 0
    assert not result.state.population.alive.any()


def test_postaction_death_moves_but_cannot_reproduce(step_case) -> None:
    state, config = step_case(
        action=ActionCode.MOVE_EAST,
        energy=0.15,
        energy_changes={"basal_cost": 0.05, "movement_cost": 0.2},
    )
    result = simulation_step(
        state=state, accumulator=create_empty_metrics_accumulator(), config=config
    )
    assert result.metrics.movement_success_count.item() == 1
    assert result.metrics.death_count.item() == 1
    assert result.metrics.birth_count.item() == 0
    assert not result.state.population.alive.any()


def test_birth_is_mutated_after_copy_and_does_not_act_in_birth_step(step_case) -> None:
    state, config = step_case(
        action=ActionCode.REPRODUCE,
        energy=60,
        energy_changes={"basal_cost": 0.01},
        evolution_changes={
            "weight_mutation_rate": 1.0,
            "weight_mutation_sigma": 0.1,
            "bias_mutation_rate": 1.0,
            "bias_mutation_sigma": 0.1,
        },
    )
    result = simulation_step(
        state=state, accumulator=create_empty_metrics_accumulator(), config=config
    )
    newborn = result.events.births.born
    child_slot = int(jnp.argmax(newborn))
    assert result.metrics.birth_count.item() == 1
    assert result.state.population.age[child_slot].item() == 0
    assert result.state.population.birth_step[child_slot].item() == 3
    assert result.state.population.parent_id[child_slot].item() == 0
    assert result.state.population.lineage_id[child_slot].item() == 0
    assert result.state.population.generation[child_slot].item() == 1
    assert result.metrics.movement_success_count.item() == 0
    assert result.metrics.feeding_success_count.item() == 0
    assert result.events.births.mutation_effective_count[child_slot].item() > 0
    assert not jnp.array_equal(
        result.state.genomes.layer1.weight[child_slot], result.state.genomes.layer1.weight[0]
    )
