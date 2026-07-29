from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core import (
    COUNT_DTYPE,
    ID_DTYPE,
    INDEX_DTYPE,
    MASK_DTYPE,
    REAL_DTYPE,
    STEP_DTYPE,
    IdCounters,
    PopulationState,
    RngState,
    SimulationState,
    WorldState,
    advance_rng,
    allocate_agent_ids,
    create_id_counters,
    create_rng_state,
)

CAPACITY = 8
HEIGHT = 4
WIDTH = 6


def _world() -> WorldState:
    return WorldState(
        resources=jnp.ones((HEIGHT, WIDTH), dtype=REAL_DTYPE),
        environment=jnp.zeros((HEIGHT, WIDTH), dtype=REAL_DTYPE),
        occupancy=jnp.full((HEIGHT, WIDTH), 3, dtype=COUNT_DTYPE),
    )


def _population(active_slots: int = 0) -> PopulationState:
    alive = jnp.arange(CAPACITY, dtype=INDEX_DTYPE) < active_slots
    position = jnp.stack(
        (
            jnp.arange(CAPACITY, dtype=INDEX_DTYPE),
            jnp.arange(CAPACITY, dtype=INDEX_DTYPE) + 10,
        ),
        axis=1,
    )
    return PopulationState(
        alive=alive.astype(MASK_DTYPE),
        agent_id=jnp.arange(CAPACITY, dtype=ID_DTYPE) + 100,
        parent_id=jnp.zeros(CAPACITY, dtype=ID_DTYPE),
        lineage_id=jnp.zeros(CAPACITY, dtype=ID_DTYPE),
        genome_id=jnp.zeros(CAPACITY, dtype=ID_DTYPE),
        generation=jnp.zeros(CAPACITY, dtype=COUNT_DTYPE),
        position=position,
        energy=jnp.arange(CAPACITY, dtype=REAL_DTYPE),
        birth_step=jnp.zeros(CAPACITY, dtype=STEP_DTYPE),
        age=jnp.zeros(CAPACITY, dtype=COUNT_DTYPE),
    )


def _state(active_slots: int = 0) -> SimulationState:
    return SimulationState(
        step=jnp.asarray(2, dtype=STEP_DTYPE),
        rng=create_rng_state(42),
        ids=create_id_counters(),
        world=_world(),
        population=_population(active_slots),
    )


def _assert_array_leaves(value: object, expected_count: int) -> None:
    leaves = jax.tree.leaves(value)
    assert len(leaves) == expected_count
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


def _population_shapes(population: PopulationState) -> tuple[tuple[int, ...], ...]:
    return tuple(leaf.shape for leaf in jax.tree.leaves(population))


def test_world_state_shapes_dtypes_and_pytree() -> None:
    world = _world()

    assert world.resources.shape == world.environment.shape == world.occupancy.shape == (4, 6)
    assert world.resources.dtype == world.environment.dtype == jnp.dtype(REAL_DTYPE)
    assert world.occupancy.dtype == jnp.dtype(COUNT_DTYPE)
    assert int(world.occupancy.max()) > 1
    _assert_array_leaves(world, 3)
    leaves, treedef = jax.tree.flatten(world)
    assert isinstance(jax.tree.unflatten(treedef, leaves), WorldState)


def test_population_state_field_contract_and_coordinate_order() -> None:
    population = _population(active_slots=2)
    expected = {
        "alive": ((8,), MASK_DTYPE),
        "agent_id": ((8,), ID_DTYPE),
        "parent_id": ((8,), ID_DTYPE),
        "lineage_id": ((8,), ID_DTYPE),
        "genome_id": ((8,), ID_DTYPE),
        "generation": ((8,), COUNT_DTYPE),
        "position": ((8, 2), INDEX_DTYPE),
        "energy": ((8,), REAL_DTYPE),
        "birth_step": ((8,), STEP_DTYPE),
        "age": ((8,), COUNT_DTYPE),
    }

    for field, (shape, dtype) in expected.items():
        array = getattr(population, field)
        assert array.shape == shape
        assert array.dtype == jnp.dtype(dtype)
    assert population.position[:, 0].tolist() == list(range(CAPACITY))
    assert population.position[:, 1].tolist() == list(range(10, 10 + CAPACITY))
    _assert_array_leaves(population, 10)


def test_population_is_structure_of_arrays_not_agents() -> None:
    population = _population(active_slots=2)
    leaves = jax.tree.leaves(population)

    assert not isinstance(population, Sequence)
    assert not any(isinstance(leaf, (list, tuple)) for leaf in leaves)
    assert not any(leaf.dtype == jnp.dtype("O") for leaf in leaves)
    assert all(leaf.shape[0] == CAPACITY for leaf in leaves)


def test_fixed_capacity_is_independent_of_living_population() -> None:
    two_alive = _population(active_slots=2)
    five_alive = _population(active_slots=5)

    assert _population_shapes(two_alive) == _population_shapes(five_alive)
    assert jax.tree.structure(two_alive) == jax.tree.structure(five_alive)
    assert two_alive.position.shape == five_alive.position.shape == (8, 2)
    assert all(leaf.shape[0] not in {2, 5} for leaf in jax.tree.leaves(five_alive))
    assert int(jnp.sum(~five_alive.alive)) == 3


def test_simulation_state_is_an_array_only_stable_pytree() -> None:
    state = _state(active_slots=1)
    other = _state(active_slots=5)

    assert state.step.shape == ()
    assert state.step.dtype == jnp.dtype(STEP_DTYPE)
    assert isinstance(state.world, WorldState)
    assert isinstance(state.population, PopulationState)
    assert isinstance(state.rng, RngState)
    assert isinstance(state.ids, IdCounters)
    _assert_array_leaves(state, 18)
    assert jax.dtypes.issubdtype(state.rng.key.dtype, jax.dtypes.prng_key)
    assert jax.tree.structure(state) == jax.tree.structure(other)
    zeroed = jax.tree.map(jnp.zeros_like, state)
    assert isinstance(zeroed, SimulationState)
    assert jax.tree.structure(zeroed) == jax.tree.structure(state)


def test_filter_jit_consumes_and_returns_state() -> None:
    def consume_eager(state: SimulationState) -> jax.Array:
        alive_energy = jnp.where(
            state.population.alive,
            state.population.energy,
            jnp.asarray(0.0, dtype=REAL_DTYPE),
        )
        return (
            state.step.astype(REAL_DTYPE) + jnp.sum(state.world.resources) + jnp.sum(alive_energy)
        )

    consume_compiled = eqx.filter_jit(consume_eager)

    @eqx.filter_jit
    def return_state(state: SimulationState) -> SimulationState:
        return state

    first = _state(active_slots=2)
    second = _state(active_slots=5)
    eager = consume_eager(first)
    compiled = consume_compiled(first)

    assert eager.shape == compiled.shape == ()
    assert jnp.array_equal(eager, compiled)
    assert float(consume_compiled(second)) > float(compiled)
    assert _population_shapes(first.population) == _population_shapes(second.population)
    returned = return_state(second)
    assert isinstance(returned, SimulationState)
    assert jax.tree.structure(returned) == jax.tree.structure(second)


def test_rng_and_id_updates_do_not_modify_ecological_state() -> None:
    state = _state(active_slots=3)
    next_rng, _ = advance_rng(state.rng)
    next_ids, _ = allocate_agent_ids(state.ids, jnp.asarray([True, False, True], dtype=MASK_DTYPE))
    rng_updated = eqx.tree_at(lambda value: value.rng, state, next_rng)
    ids_updated = eqx.tree_at(lambda value: value.ids, state, next_ids)

    for updated in (rng_updated, ids_updated):
        assert jax.tree.all(jax.tree.map(jnp.array_equal, state.world, updated.world))
        assert jax.tree.all(jax.tree.map(jnp.array_equal, state.population, updated.population))


def test_in_memory_state_is_sufficient_for_deterministic_continuation() -> None:
    state = _state(active_slots=2)
    rng, _ = advance_rng(state.rng)
    intermediate = eqx.tree_at(lambda value: value.rng, state, rng)

    def continue_branch(value: SimulationState) -> tuple[jax.Array, jax.Array, jax.Array]:
        next_rng, step_key = advance_rng(value.rng)
        next_ids, allocation = allocate_agent_ids(
            value.ids, jnp.asarray([False, True, True], dtype=MASK_DTYPE)
        )
        return next_rng.key, step_key, next_ids.next_agent_id + allocation.count

    first = continue_branch(intermediate)
    second = continue_branch(intermediate)
    assert all(jnp.array_equal(left, right) for left, right in zip(first, second, strict=True))
