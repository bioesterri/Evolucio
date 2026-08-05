from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from evolucio.config import ExperimentConfig, compile_config, load_config
from evolucio.core import (
    COUNT_DTYPE,
    ID_DTYPE,
    INDEX_DTYPE,
    MASK_DTYPE,
    MAX_NEXT_ID,
    NULL_ID,
    REAL_DTYPE,
    STEP_DTYPE,
    IdCounters,
    PopulationState,
    WorldState,
    create_id_counters,
    create_rng_state,
)
from evolucio.core.population import (
    INACTIVE_POSITION_COORDINATE,
    PopulationInitializationResult,
    create_empty_population,
    initialize_population,
)
from evolucio.core.population.init import _build_initial_alive_mask
from evolucio.core.world import compute_occupancy, initialize_world


@pytest.fixture
def config() -> ExperimentConfig:
    return load_config("tests/fixtures/config/valid_v1.yaml")


def changed(config: ExperimentConfig, *, population: dict[str, object]) -> ExperimentConfig:
    raw = config.model_dump(mode="python")
    raw["population"].update(population)  # type: ignore[union-attr]
    return ExperimentConfig.model_validate(raw)


def result(
    config: ExperimentConfig,
    *,
    initial_agents: int = 3,
    max_agents: int = 8,
    seed: int = 9,
    ids: IdCounters | None = None,
) -> PopulationInitializationResult:
    host = changed(
        config,
        population={
            "initial_agents": initial_agents,
            "max_agents": max_agents,
            "max_births_per_step": min(max_agents, 2),
        },
    )
    core = compile_config(host).core
    world = initialize_world(core.world, create_rng_state(seed).key)
    return initialize_population(
        world,
        core.population,
        core.energy,
        create_rng_state(seed).key,
        ids or create_id_counters(),
    )


def test_empty_population_is_canonical_fixed_shape_pytree() -> None:
    population = create_empty_population(4)
    expected = {
        "alive": ((4,), MASK_DTYPE),
        "agent_id": ((4,), ID_DTYPE),
        "parent_id": ((4,), ID_DTYPE),
        "lineage_id": ((4,), ID_DTYPE),
        "genome_id": ((4,), ID_DTYPE),
        "generation": ((4,), COUNT_DTYPE),
        "position": ((4, 2), INDEX_DTYPE),
        "energy": ((4,), REAL_DTYPE),
        "birth_step": ((4,), STEP_DTYPE),
        "age": ((4,), COUNT_DTYPE),
    }
    for name, (shape, dtype) in expected.items():
        leaf = getattr(population, name)
        assert isinstance(leaf, jax.Array) and leaf.shape == shape
        assert leaf.dtype == jnp.dtype(dtype)
    assert not bool(jnp.any(population.alive))
    assert bool(jnp.all(population.position == INACTIVE_POSITION_COORDINATE))
    assert bool(jnp.all(population.energy == 0))
    for ids in (
        population.agent_id,
        population.parent_id,
        population.lineage_id,
        population.genome_id,
    ):
        assert bool(jnp.all(ids == NULL_ID))
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(population))


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, [False] * 8), (3, [True] * 3 + [False] * 5), (8, [True] * 8)],
)
def test_alive_mask_eager_and_jit(count: int, expected: list[bool]) -> None:
    value = jnp.asarray(count, dtype=STEP_DTYPE)
    eager = _build_initial_alive_mask(initial_agents=value, max_agents=8)
    compiled = jax.jit(_build_initial_alive_mask, static_argnames=("max_agents",))(
        initial_agents=value, max_agents=8
    )
    assert eager.tolist() == expected
    assert eager.shape == (8,) and eager.dtype == jnp.dtype(MASK_DTYPE)
    assert jnp.array_equal(eager, compiled)


def test_founder_ids_genealogy_and_physiology(config: ExperimentConfig) -> None:
    initialized = result(
        config,
        ids=create_id_counters(next_agent_id=10, next_genome_id=100, next_lineage_id=1000),
    )
    population = initialized.population
    assert population.agent_id.tolist() == [10, 11, 12, -1, -1, -1, -1, -1]
    assert population.genome_id.tolist() == [100, 101, 102, -1, -1, -1, -1, -1]
    assert population.lineage_id.tolist() == [1000, 1001, 1002, -1, -1, -1, -1, -1]
    assert bool(jnp.all(population.parent_id == NULL_ID))
    assert initialized.ids.next_agent_id == 13
    assert initialized.ids.next_genome_id == 103
    assert initialized.ids.next_lineage_id == 1003
    assert bool(jnp.all(population.energy[:3] == 20.0))
    assert bool(jnp.all(population.energy[3:] == 0.0))
    assert bool(jnp.all(population.age == 0))
    assert bool(jnp.all(population.generation == 0))
    assert bool(jnp.all(population.birth_step == 0))


@pytest.mark.parametrize("domain", ["agent", "genome", "lineage"])
def test_identifier_overflow_is_atomic(config: ExperimentConfig, domain: str) -> None:
    values = {"next_agent_id": 10, "next_genome_id": 100, "next_lineage_id": 1000}
    values[f"next_{domain}_id"] = MAX_NEXT_ID
    ids = create_id_counters(**values)  # type: ignore[arg-type]
    core = compile_config(changed(config, population={"initial_agents": 2})).core
    world = initialize_world(core.world, create_rng_state(2).key)
    initialized = initialize_population(
        world, core.population, core.energy, create_rng_state(2).key, ids
    )
    assert bool(initialized.overflow)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, initialized.ids, ids))
    assert jax.tree.all(jax.tree.map(jnp.array_equal, initialized.world, world))
    assert not bool(jnp.any(initialized.population.alive))
    assert bool(jnp.all(initialized.population.agent_id == NULL_ID))
    assert bool(jnp.all(initialized.population.genome_id == NULL_ID))
    assert bool(jnp.all(initialized.population.lineage_id == NULL_ID))


def test_positions_overlap_occupancy_and_world_conservation(config: ExperimentConfig) -> None:
    raw = config.model_dump(mode="python")
    raw["world"].update({"width": 1, "height": 1})  # type: ignore[union-attr]
    raw["population"].update(  # type: ignore[union-attr]
        {"initial_agents": 4, "max_agents": 6, "max_births_per_step": 2}
    )
    core = compile_config(ExperimentConfig.model_validate(raw)).core
    world = initialize_world(core.world, create_rng_state(4).key)
    initialized = initialize_population(
        world, core.population, core.energy, create_rng_state(4).key, create_id_counters()
    )
    assert initialized.population.position[:4].tolist() == [[0, 0]] * 4
    assert initialized.population.position[4:].tolist() == [[-1, -1]] * 2
    assert initialized.world.occupancy.shape == (1, 1)
    assert initialized.world.occupancy.dtype == jnp.dtype(COUNT_DTYPE)
    assert int(initialized.world.occupancy[0, 0]) == 4
    assert jnp.array_equal(initialized.world.resources, world.resources)
    assert jnp.array_equal(initialized.world.environment, world.environment)
    assert not bool(initialized.overflow)


def test_zero_capacity_full_capacity_determinism_and_keys(config: ExperimentConfig) -> None:
    zero = result(config, initial_agents=0)
    full = result(config, initial_agents=8)
    same = result(config, initial_agents=8)
    other = result(config, initial_agents=8, seed=123)
    assert int(zero.world.occupancy.sum()) == 0 and not bool(zero.overflow)
    assert bool(jnp.all(full.population.alive)) and int(full.world.occupancy.sum()) == 8
    assert jnp.array_equal(full.population.position, same.population.position)
    assert not jnp.array_equal(full.population.position, other.population.position)


def test_initial_agents_reveals_presampled_positions_and_preserves_signature(
    config: ExperimentConfig,
) -> None:
    host_three = changed(
        config, population={"initial_agents": 3, "max_agents": 8, "max_births_per_step": 2}
    )
    host_five = changed(
        config, population={"initial_agents": 5, "max_agents": 8, "max_births_per_step": 2}
    )
    three = result(config, initial_agents=3)
    five = result(config, initial_agents=5)
    assert three.population.position.shape == five.population.position.shape == (8, 2)
    assert jnp.array_equal(three.population.position[:3], five.population.position[:3])
    assert jax.tree.structure(three) == jax.tree.structure(five)
    assert (
        compile_config(host_three).compile_signature == compile_config(host_five).compile_signature
    )
    assert compile_config(host_three).config_hash != compile_config(host_five).config_hash


def test_resources_environment_and_stream_order_do_not_affect_positions(
    config: ExperimentConfig,
) -> None:
    baseline = result(config)
    raw = config.model_dump(mode="python")
    raw["world"].update(  # type: ignore[union-attr]
        {
            "resource_distribution": "uniform",
            "initial_resource_mean": 1.0,
            "environment_initial_value": 0.8,
        }
    )
    altered = result(ExperimentConfig.model_validate(raw))
    assert jnp.array_equal(baseline.population.position, altered.population.position)


def test_occupancy_manual_coordinate_order_and_inactive_weight() -> None:
    positions = jnp.asarray([[2, 0], [0, 1], [2, 0], [-1, -1]], dtype=INDEX_DTYPE)
    alive = jnp.asarray([True, True, True, False], dtype=MASK_DTYPE)
    population = eqx.tree_at(
        lambda state: (state.position, state.alive),
        create_empty_population(4),
        (positions, alive),
    )
    eager = compute_occupancy(population, width=3, height=2)
    compiled = eqx.filter_jit(compute_occupancy)(population, width=3, height=2)
    assert eager.occupancy.tolist() == [[0, 0, 2], [1, 0, 0]]
    assert jnp.array_equal(eager.occupancy, compiled.occupancy)


def test_initialize_population_eager_jit_and_scan(config: ExperimentConfig) -> None:
    host = changed(
        config, population={"initial_agents": 3, "max_agents": 8, "max_births_per_step": 2}
    )
    core = compile_config(host).core
    key = create_rng_state(7).key
    world = initialize_world(core.world, key)
    ids = create_id_counters()
    eager = initialize_population(world, core.population, core.energy, key, ids)
    compiled = eqx.filter_jit(initialize_population)(world, core.population, core.energy, key, ids)
    assert isinstance(eager, PopulationInitializationResult)
    assert isinstance(eager.population, PopulationState) and isinstance(eager.world, WorldState)
    assert eager.overflow.shape == () and eager.overflow.dtype == jnp.dtype(MASK_DTYPE)
    assert jax.tree.structure(eager) == jax.tree.structure(compiled)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, eager, compiled))

    def body(
        state: PopulationInitializationResult, _: jax.Array
    ) -> tuple[PopulationInitializationResult, jax.Array]:
        return state, state.world.occupancy.sum()

    scanned, totals = jax.lax.scan(body, eager, jnp.arange(2))
    assert jax.tree.structure(scanned) == jax.tree.structure(eager)
    assert totals.tolist() == [3, 3]


@given(
    max_agents=st.integers(1, 12),
    initial_agents=st.integers(0, 12),
    width=st.integers(1, 5),
    height=st.integers(1, 5),
    seed=st.integers(0, 1000),
)
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_small_valid_population_properties(
    config: ExperimentConfig,
    max_agents: int,
    initial_agents: int,
    width: int,
    height: int,
    seed: int,
) -> None:
    initial_agents = min(initial_agents, max_agents)
    raw = config.model_dump(mode="python")
    raw["world"].update({"width": width, "height": height})  # type: ignore[union-attr]
    raw["population"].update(  # type: ignore[union-attr]
        {"initial_agents": initial_agents, "max_agents": max_agents, "max_births_per_step": 1}
    )
    host = ExperimentConfig.model_validate(raw)
    initialized = result(host, initial_agents=initial_agents, max_agents=max_agents, seed=seed)
    population = initialized.population
    assert int(population.alive.sum()) == initial_agents
    assert population.position.shape == (max_agents, 2)
    assert int(initialized.world.occupancy.sum()) == initial_agents
    assert bool(jnp.all(population.position[population.alive, 0] < width))
    assert bool(jnp.all(population.position[population.alive, 1] < height))
    assert bool(jnp.all(population.position[~population.alive] == -1))
    assert len(set(population.agent_id[population.alive].tolist())) == initial_agents
    assert bool(jnp.all(jnp.isfinite(population.energy)))
    same = result(host, initial_agents=initial_agents, max_agents=max_agents, seed=seed)
    assert jnp.array_equal(population.position, same.population.position)


def test_new_module_contains_no_prohibited_random_or_agent_patterns() -> None:
    source = Path("src/evolucio/core/population/init.py").read_text()
    for prohibited in (
        "numpy.random",
        "np.random",
        "import random",
        "jax.random.PRNGKey",
        "uuid",
        "list[Agent]",
    ):
        assert prohibited not in source
