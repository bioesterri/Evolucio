import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.actions import (
    FeedingResolutionCode,
    FeedingResolutionResult,
    resolve_feeding,
)
from evolucio.core.codes import ActionCode
from evolucio.core.state import PopulationState, WorldState


def states(
    energy: list[float], resources: list[list[float]], positions: list[list[int]] | None = None
) -> tuple[PopulationState, WorldState]:
    capacity = len(energy)
    position = positions or [[0, 0] for _ in energy]
    zeros = jnp.zeros((capacity,), dtype=jnp.int32)
    population = PopulationState(
        alive=jnp.ones((capacity,), dtype=jnp.bool_),
        agent_id=jnp.arange(capacity, dtype=jnp.int32),
        parent_id=zeros,
        lineage_id=zeros,
        genome_id=zeros,
        generation=zeros,
        position=jnp.asarray(position, dtype=jnp.int32),
        energy=jnp.asarray(energy, dtype=jnp.float32),
        birth_step=zeros,
        age=zeros,
    )
    shape = (len(resources), len(resources[0]))
    world = WorldState(
        resources=jnp.asarray(resources, dtype=jnp.float32),
        environment=jnp.zeros(shape, dtype=jnp.float32),
        occupancy=jnp.zeros(shape, dtype=jnp.int32),
    )
    return population, world


def feed(
    population: PopulationState, world: WorldState, actions: list[int]
) -> FeedingResolutionResult:
    return resolve_feeding(
        population=population,
        world=world,
        actions_after_movement=jnp.asarray(actions, dtype=jnp.int32),
        maximum_energy=jnp.asarray(10.0, dtype=jnp.float32),
        energy_gain_per_resource=jnp.asarray(2.0, dtype=jnp.float32),
        feeding_max_resource_intake=jnp.asarray(3.0, dtype=jnp.float32),
        width=world.resources.shape[1],
        height=world.resources.shape[0],
    )


def test_complete_feeding_and_no_eat_preserves_action() -> None:
    population, world = states([2, 4], [[5]], [[0, 0], [0, 0]])
    result = feed(population, world, [ActionCode.EAT, ActionCode.STAY])
    assert result.feeding_codes.tolist() == [FeedingResolutionCode.FED_FULL, 0]
    assert result.resource_consumed.tolist() == [3, 0]
    assert result.population.energy.tolist() == [8, 4]
    assert result.world.resources.tolist() == [[2]]
    assert result.actions_after_feeding.tolist() == [ActionCode.EAT, ActionCode.STAY]


def test_headroom_no_resource_and_maximum() -> None:
    population, world = states([9, 10], [[1], [0]], [[0, 0], [0, 1]])
    result = feed(population, world, [ActionCode.EAT, ActionCode.EAT])
    assert result.resource_consumed.tolist() == [0.5, 0]
    assert result.population.energy.tolist() == [10, 10]
    assert result.feeding_codes.tolist() == [
        FeedingResolutionCode.FED_FULL,
        FeedingResolutionCode.NO_RESOURCE,
    ]
    assert result.actions_after_feeding.tolist() == [ActionCode.EAT, ActionCode.STAY]


def test_invalid_inputs_are_safe() -> None:
    population, world = states([jnp.nan], [[2]], [[-1, 0]])
    result = feed(population, world, [ActionCode.EAT])
    assert result.feeding_codes.item() == FeedingResolutionCode.INVALID_FEEDING_INPUT
    assert result.invalid_feeding_input_count.item() == 1
    assert result.resource_consumed.item() == 0
    assert result.world.resources.tolist() == [[2]]


def test_eager_jit_and_scan_have_fixed_shapes() -> None:
    population, world = states([2, 2], [[2]], [[0, 0], [0, 0]])
    actions = jnp.asarray([ActionCode.EAT, ActionCode.EAT], dtype=jnp.int32)
    kwargs = dict(
        maximum_energy=jnp.asarray(10, dtype=jnp.float32),
        energy_gain_per_resource=jnp.asarray(2, dtype=jnp.float32),
        feeding_max_resource_intake=jnp.asarray(3, dtype=jnp.float32),
        width=1,
        height=1,
    )
    eager = resolve_feeding(
        population=population, world=world, actions_after_movement=actions, **kwargs
    )
    compiled = eqx.filter_jit(resolve_feeding)(
        population=population, world=world, actions_after_movement=actions, **kwargs
    )
    assert jnp.allclose(eager.resource_consumed, compiled.resource_consumed)

    def body(
        carry: tuple[PopulationState, WorldState], _: None
    ) -> tuple[tuple[PopulationState, WorldState], jax.Array]:
        result = resolve_feeding(
            population=carry[0], world=carry[1], actions_after_movement=actions, **kwargs
        )
        return (result.population, result.world), result.resource_consumed

    (_, scanned_world), consumed = jax.lax.scan(body, (population, world), xs=None, length=1)
    assert consumed.shape == (1, 2)
    assert scanned_world.resources.shape == (1, 1)
