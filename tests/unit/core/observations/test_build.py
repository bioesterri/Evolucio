import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from evolucio.config import CoreConfig, ExperimentConfig, compile_config
from evolucio.core import (
    COUNT_DTYPE,
    ID_DTYPE,
    INDEX_DTYPE,
    MASK_DTYPE,
    REAL_DTYPE,
    STEP_DTYPE,
    PopulationState,
    SimulationState,
    WorldState,
    create_id_counters,
    create_rng_state,
)
from evolucio.core.observations import OBSERVATION_SIZE, ObservationIndex, build_observations


def _setup(config: ExperimentConfig) -> tuple[SimulationState, CoreConfig]:
    raw = config.model_dump(mode="python")
    raw["world"].update(
        width=5,
        height=5,
        resource_capacity=10.0,
        initial_resource_mean=0.0,
        resource_distribution="uniform",
    )
    raw["population"].update(initial_agents=4, max_agents=4, max_births_per_step=2)
    raw["observations"].update(perception_radius=1)
    cfg = compile_config(ExperimentConfig.model_validate(raw)).core
    resources = (
        jnp.zeros((5, 5), dtype=REAL_DTYPE)
        .at[2, 2]
        .set(5)
        .at[1, 2]
        .set(10)
        .at[3, 2]
        .set(5)
        .at[2, 3]
        .set(2)
        .at[2, 1]
        .set(8)
    )
    environment = jnp.zeros((5, 5), dtype=REAL_DTYPE).at[2, 2].set(0.25)
    positions = jnp.asarray([[2, 2], [2, 2], [2, 1], [4, 4]], dtype=INDEX_DTYPE)
    alive = jnp.asarray([True, True, True, False], dtype=MASK_DTYPE)
    occupancy = jnp.zeros((5, 5), dtype=COUNT_DTYPE).at[2, 2].set(2).at[1, 2].set(1)
    zeros_id = jnp.zeros(4, dtype=ID_DTYPE)
    population = PopulationState(
        alive=alive,
        agent_id=zeros_id,
        parent_id=zeros_id,
        lineage_id=zeros_id,
        genome_id=zeros_id,
        generation=jnp.zeros(4, dtype=COUNT_DTYPE),
        position=positions,
        energy=jnp.asarray([50, 40, 20, 99], dtype=REAL_DTYPE),
        birth_step=jnp.zeros(4, dtype=STEP_DTYPE),
        age=jnp.asarray([500, 0, 100, 999], dtype=COUNT_DTYPE),
    )
    state = SimulationState(
        step=jnp.asarray(7, dtype=STEP_DTYPE),
        rng=create_rng_state(1),
        ids=create_id_counters(),
        world=WorldState(resources=resources, environment=environment, occupancy=occupancy),
        population=population,
    )
    return state, cfg


def test_complete_vector_manual_masking_and_jit(config: ExperimentConfig) -> None:
    state, cfg = _setup(config)
    result = build_observations(state, cfg)
    expected = jnp.asarray(
        [0.5, 0.5, 0.1, 0.5, 1.0, 0.5, 0.2, 0.8, 0.25, 0.0, 0.0, 0.0, 0.5, 0.25, 0.0],
        dtype=REAL_DTYPE,
    )
    assert result.shape == (4, OBSERVATION_SIZE)
    assert result.dtype == jnp.dtype(REAL_DTYPE)
    assert jnp.allclose(result[0], expected)
    assert jnp.array_equal(result[3], jnp.zeros(15, dtype=REAL_DTYPE))
    assert jnp.allclose(eqx.filter_jit(build_observations)(state, cfg), result)


def test_invalid_alive_position_is_entirely_zero(config: ExperimentConfig) -> None:
    state, cfg = _setup(config)
    state = eqx.tree_at(
        lambda value: value.population.position,
        state,
        state.population.position.at[0].set(jnp.asarray([-1, 2], dtype=INDEX_DTYPE)),
    )
    result = eqx.filter_jit(build_observations)(state, cfg)
    assert jnp.array_equal(result[0], jnp.zeros(15, dtype=REAL_DTYPE))
    assert jnp.all(jnp.isfinite(result))


@pytest.mark.parametrize(
    "position,expected",
    [
        ((2, 2), 0.0),
        ((2, 0), 1 / 15),
        ((2, 4), 2 / 15),
        ((4, 2), 4 / 15),
        ((0, 2), 8 / 15),
        ((0, 0), 9 / 15),
    ],
)
def test_blocked_direction_identity(
    config: ExperimentConfig, position: tuple[int, int], expected: float
) -> None:
    state, cfg = _setup(config)
    positions = state.population.position.at[0].set(jnp.asarray(position, dtype=INDEX_DTYPE))
    state = eqx.tree_at(lambda value: value.population.position, state, positions)
    assert float(
        build_observations(state, cfg)[0, ObservationIndex.MOVEMENT_BLOCKED]
    ) == pytest.approx(expected)


def test_locality_identity_independence_and_purity(config: ExperimentConfig) -> None:
    state, cfg = _setup(config)
    before = build_observations(state, cfg)[0]
    changed = eqx.tree_at(
        lambda value: value.world.resources, state, state.world.resources.at[4, 4].set(10)
    )
    changed = eqx.tree_at(
        lambda value: value.population.agent_id, changed, changed.population.agent_id + 100
    )
    assert jnp.array_equal(build_observations(changed, cfg)[0], before)
    assert jnp.array_equal(build_observations(state, cfg), build_observations(state, cfg))
    assert int(state.step) == 7 and float(state.population.energy[0]) == 50


def test_fixed_capacity_zero_full_and_scan(config: ExperimentConfig) -> None:
    state, cfg = _setup(config)
    eager = jnp.stack([build_observations(state, cfg) for _ in range(3)])

    def body(carry: SimulationState, _: None) -> tuple[SimulationState, jax.Array]:
        return carry, build_observations(carry, cfg)

    _, scanned = jax.lax.scan(body, state, None, length=3)
    assert scanned.shape == (3, 4, 15)
    assert jnp.allclose(scanned, eager)
    for alive in (jnp.zeros(4, dtype=MASK_DTYPE), jnp.ones(4, dtype=MASK_DTYPE)):
        changed = eqx.tree_at(lambda value: value.population.alive, state, alive)
        assert build_observations(changed, cfg).shape == (4, 15)
