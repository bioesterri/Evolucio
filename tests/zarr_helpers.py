"""Non-trivial checkpoint scenario with conflict, births, mutation and deaths."""

# pyright: reportUnknownMemberType=false
import equinox as eqx
import jax.numpy as jnp

from evolucio.config import ExperimentConfig, load_config
from evolucio.config.compile import CompiledConfig, compile_config
from evolucio.core.codes import ActionCode
from evolucio.core.ids import create_id_counters
from evolucio.core.policy import create_empty_genome_batch
from evolucio.core.population import create_empty_population
from evolucio.core.rng import create_rng_state
from evolucio.core.spatial import rebuild_world_occupancy
from evolucio.core.state import SimulationState
from evolucio.core.world import initialize_world


def checkpoint_host_config() -> ExperimentConfig:
    data = load_config("tests/fixtures/config/valid_v1.json").model_dump()
    data["seed"] = 1729
    data["world"].update(width=6, height=6, resource_distribution="uniform")
    data["population"].update(
        initial_agents=0, max_agents=16, max_births_per_step=4, allow_multiple_agents_per_cell=False
    )
    data["energy"].update(
        initial_energy=60.0, basal_cost=0.1, movement_cost=0.05, reproduction_threshold=25.0
    )
    data["evolution"].update(
        min_reproduction_age=1,
        max_age=70,
        weight_mutation_rate=1.0,
        bias_mutation_rate=1.0,
        weight_mutation_sigma=0.05,
        bias_mutation_sigma=0.05,
    )
    return ExperimentConfig.model_validate(data)


def checkpoint_scenario() -> tuple[SimulationState, CompiledConfig]:
    prepared = compile_config(checkpoint_host_config())
    rng = create_rng_state(1729)
    population = create_empty_population(16)
    updates = {
        "alive": population.alive.at[:5].set(True),
        "agent_id": population.agent_id.at[:5].set(jnp.arange(5)),
        "genome_id": population.genome_id.at[:5].set(jnp.arange(5)),
        "lineage_id": population.lineage_id.at[:5].set(jnp.arange(5)),
        "position": population.position.at[:5].set(
            jnp.asarray([[4, 4], [2, 2], [1, 3], [3, 3], [0, 0]])
        ),
        "energy": population.energy.at[:5].set(jnp.asarray([60.0, 20.0, 20.0, 20.0, 0.05])),
    }
    for name, value in updates.items():
        population = eqx.tree_at(lambda p, name=name: getattr(p, name), population, value)
    world = initialize_world(prepared.core.world, rng.key)
    world = rebuild_world_occupancy(world, population, width=6, height=6).world
    genomes = create_empty_genome_batch(16)
    biases = genomes.layer2.bias
    for slot, action in enumerate(
        (
            ActionCode.REPRODUCE,
            ActionCode.EAT,
            ActionCode.MOVE_EAST,
            ActionCode.MOVE_WEST,
            ActionCode.STAY,
        )
    ):
        biases = biases.at[slot, int(action)].set(2)
    genomes = eqx.tree_at(lambda g: g.layer2.bias, genomes, biases)
    state = SimulationState(
        step=jnp.asarray(0, jnp.int32),
        rng=rng,
        ids=create_id_counters(next_agent_id=5, next_genome_id=5, next_lineage_id=5),
        world=world,
        population=population,
        genomes=genomes,
    )
    return state, prepared


def assert_checkpoint_bitwise_equal(actual: object, expected: object) -> None:
    """Compare carry bytes, including signed zero and the typed key's raw data."""
    import jax
    import numpy as np

    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert left.dtype == right.dtype and left.shape == right.shape
        if jax.dtypes.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        a, b = np.asarray(jax.device_get(left)), np.asarray(jax.device_get(right))
        assert a.tobytes() == b.tobytes()
