import ast
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.config import ExperimentConfig, compile_config, load_config
from evolucio.core import COUNT_DTYPE, REAL_DTYPE, STEP_DTYPE, WorldState
from evolucio.core.world import update_world_for_step


def configured_world() -> tuple[object, WorldState]:
    host = load_config("tests/fixtures/config/valid_v1.yaml")
    raw = host.model_dump(mode="python")
    raw["world"].update(  # type: ignore[union-attr]
        {
            "resource_capacity": 1.0,
            "initial_resource_mean": 0.5,
            "regeneration_rate": 0.2,
            "environment_initial_value": 0.1,
            "environment_schedule": [
                {
                    "start_step": 0,
                    "end_step": 1,
                    "regeneration_multiplier": 0.5,
                    "stress_level": 0.8,
                },
                {
                    "start_step": 2,
                    "end_step": 3,
                    "regeneration_multiplier": 0.0,
                    "stress_level": 0.6,
                },
            ],
        }
    )
    config = compile_config(ExperimentConfig.model_validate(raw)).core.world
    world = WorldState(
        resources=jnp.asarray([[0.0, 0.5]], dtype=REAL_DTYPE),
        environment=jnp.zeros((1, 2), dtype=REAL_DTYPE),
        occupancy=jnp.asarray([[1, 2]], dtype=COUNT_DTYPE),
    )
    return config, world


def test_step_zero_updates_environment_regenerates_resources_and_preserves_occupancy() -> None:
    config, world = configured_world()
    updated = update_world_for_step(world, jnp.asarray(0, STEP_DTYPE), config)  # type: ignore[arg-type]
    assert isinstance(updated, WorldState)
    assert jnp.allclose(updated.resources, jnp.asarray([[0.1, 0.55]], dtype=REAL_DTYPE))
    assert jnp.allclose(updated.environment, 0.8)
    assert jnp.array_equal(updated.occupancy, world.occupancy)


def test_later_steps_regenerate_and_jit_matches() -> None:
    config, world = configured_world()
    step = jnp.asarray(1, STEP_DTYPE)
    eager = update_world_for_step(world, step, config)  # type: ignore[arg-type]
    compiled = eqx.filter_jit(update_world_for_step)(world, step, config)
    assert jnp.allclose(eager.resources, jnp.asarray([[0.2, 0.6]], dtype=REAL_DTYPE))
    assert jnp.allclose(eager.environment, 0.1)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, eager, compiled))
    stopped = update_world_for_step(world, jnp.asarray(2, STEP_DTYPE), config)  # type: ignore[arg-type]
    assert jnp.array_equal(stopped.resources, world.resources)
    assert jnp.array_equal(stopped.occupancy, world.occupancy)


def test_lax_scan_matches_eager_loop() -> None:
    config, initial = configured_world()
    steps = jnp.arange(4, dtype=STEP_DTYPE)

    def body(world: WorldState, step: jax.Array) -> tuple[WorldState, jax.Array]:
        updated = update_world_for_step(world, step, config)  # type: ignore[arg-type]
        return updated, updated.environment[0, 0]

    scanned, environments = jax.lax.scan(body, initial, steps)
    eager = initial
    eager_values = []
    for step in steps:
        eager, value = body(eager, step)
        eager_values.append(value)
    assert jax.tree.all(jax.tree.map(jnp.allclose, scanned, eager))
    assert jnp.allclose(environments, jnp.stack(eager_values))
    assert jnp.allclose(environments, jnp.asarray([0.8, 0.1, 0.6, 0.1], dtype=REAL_DTYPE))


def test_new_world_modules_have_no_random_sources() -> None:
    forbidden = ("jax.random", "random.", "numpy.random", "np.random", "core.rng")
    for path in (
        Path("src/evolucio/core/world") / name
        for name in ("environment.py", "resources.py", "update.py")
    ):
        source = path.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden)
        ast.parse(source)
