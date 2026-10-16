"""Shared deterministic integration and regression test helpers."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.config import ExperimentConfig, freeze_config, load_config
from evolucio.config.compile import CoreConfig, compile_config
from evolucio.core.codes import ActionCode
from evolucio.core.ids import create_id_counters
from evolucio.core.policy import create_empty_genome_batch
from evolucio.core.population import create_empty_population
from evolucio.core.rng import create_rng_state
from evolucio.core.spatial import rebuild_world_occupancy
from evolucio.core.state import SimulationState
from evolucio.core.world import initialize_world

FLOAT32_ATOL = 1e-6
FLOAT32_RTOL = 1e-6
REGRESSION_SEED = 1729
REGRESSION_STEPS = 8
FIXTURE_PATH = Path("tests/regression/fixtures/fixed_seed_v1.json")


def assert_pytree_equivalent(
    actual: object,
    expected: object,
    *,
    float_atol: float = FLOAT32_ATOL,
    float_rtol: float = FLOAT32_RTOL,
) -> None:
    """Compare PyTrees with exact discrete and strict finite float semantics."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    actual_leaves, _ = jax.tree.flatten(actual)
    expected_leaves, _ = jax.tree.flatten(expected)
    for index, (actual_leaf, expected_leaf) in enumerate(
        zip(actual_leaves, expected_leaves, strict=True)
    ):
        assert actual_leaf.shape == expected_leaf.shape, f"leaf {index}: shape differs"
        assert actual_leaf.dtype == expected_leaf.dtype, f"leaf {index}: dtype differs"
        if jnp.issubdtype(actual_leaf.dtype, jnp.inexact):
            assert bool(jnp.all(jnp.isfinite(actual_leaf))), f"leaf {index}: non-finite actual"
            assert bool(jnp.all(jnp.isfinite(expected_leaf))), f"leaf {index}: non-finite expected"
            difference = float(jnp.max(jnp.abs(actual_leaf - expected_leaf), initial=0))
            assert bool(
                jnp.allclose(actual_leaf, expected_leaf, atol=float_atol, rtol=float_rtol)
            ), f"leaf {index}: max abs diff {difference}"
        else:
            assert bool(jnp.array_equal(actual_leaf, expected_leaf)), (
                f"leaf {index}: expected {expected_leaf}, actual {actual_leaf}"
            )


def canonical_case(seed: int = REGRESSION_SEED) -> tuple[SimulationState, CoreConfig, str]:
    """Build the small non-trivial canonical state independently from one seed."""
    host_data = load_config("tests/fixtures/config/valid_v1.yaml").model_dump(mode="python")
    host_data["seed"] = seed
    host_data["world"].update(  # type: ignore[union-attr]
        {
            "width": 6,
            "height": 6,
            "resource_distribution": "uniform",
            "environment_schedule": [
                {
                    "start_step": 0,
                    "end_step": 4,
                    "regeneration_multiplier": 1.0,
                    "stress_level": 0.1,
                },
                {
                    "start_step": 4,
                    "end_step": 20,
                    "regeneration_multiplier": 0.5,
                    "stress_level": 0.3,
                },
            ],
        }
    )
    host_data["population"].update(  # type: ignore[union-attr]
        {"initial_agents": 0, "max_agents": 8, "max_births_per_step": 2}
    )
    host_data["energy"].update(  # type: ignore[union-attr]
        {"basal_cost": 0.1, "movement_cost": 0.05, "feeding_cost": 0.02}
    )
    host_data["evolution"].update(  # type: ignore[union-attr]
        {
            "min_reproduction_age": 1,
            "weight_mutation_rate": 0.2,
            "weight_mutation_sigma": 0.05,
            "bias_mutation_rate": 0.2,
            "bias_mutation_sigma": 0.05,
        }
    )
    host = ExperimentConfig.model_validate(host_data)
    config = compile_config(host).core
    rng = create_rng_state(seed)
    world = initialize_world(config.world, rng.key)
    population = create_empty_population(8)
    population = eqx.tree_at(
        lambda value: (
            value.alive,
            value.agent_id,
            value.lineage_id,
            value.genome_id,
            value.position,
            value.energy,
            value.age,
        ),
        population,
        (
            jnp.asarray([True, True, True, False, False, False, False, False]),
            jnp.asarray([0, 1, 2, -1, -1, -1, -1, -1], dtype=jnp.int32),
            jnp.asarray([0, 1, 2, -1, -1, -1, -1, -1], dtype=jnp.int32),
            jnp.asarray([0, 1, 2, -1, -1, -1, -1, -1], dtype=jnp.int32),
            jnp.asarray(
                [[2, 2], [1, 1], [3, 2], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
                dtype=jnp.int32,
            ),
            jnp.asarray([60.0, 20.0, 0.15, 0, 0, 0, 0, 0], dtype=jnp.float32),
            jnp.asarray([2, 2, 2, 0, 0, 0, 0, 0], dtype=jnp.int32),
        ),
    )
    world = rebuild_world_occupancy(world, population, width=6, height=6).world
    genomes = create_empty_genome_batch(8)
    biases = genomes.layer2.bias
    biases = biases.at[0, int(ActionCode.REPRODUCE)].set(2)
    biases = biases.at[1, int(ActionCode.EAT)].set(2)
    biases = biases.at[2, int(ActionCode.MOVE_WEST)].set(2)
    genomes = eqx.tree_at(lambda value: value.layer2.bias, genomes, biases)
    state = SimulationState(
        step=jnp.asarray(0, dtype=jnp.int32),
        rng=rng,
        ids=create_id_counters(next_agent_id=3, next_genome_id=3, next_lineage_id=3),
        world=world,
        population=population,
        genomes=genomes,
    )
    return state, config, freeze_config(host).config_hash


def first_divergent_step(actual: object, expected: object) -> int | None:
    """Return the first differing temporal index for diagnostic failures."""
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if actual_leaf.ndim == 0:
            continue
        equal = jnp.isclose(actual_leaf, expected_leaf, atol=FLOAT32_ATOL, rtol=FLOAT32_RTOL)
        unequal_by_step = ~jnp.all(equal.reshape((equal.shape[0], -1)), axis=1)
        if bool(jnp.any(unequal_by_step)):
            return int(jnp.argmax(unequal_by_step))
    return None
