# ruff: noqa: ANN201, ANN202

import equinox as eqx
import jax.numpy as jnp
import pytest

from evolucio.config import ExperimentConfig, compile_config, load_config
from evolucio.core import SimulationState, create_id_counters, create_rng_state
from evolucio.core.codes import ActionCode
from evolucio.core.policy import create_empty_genome_batch
from evolucio.core.population import create_empty_population
from evolucio.core.spatial import rebuild_world_occupancy
from evolucio.core.world import initialize_world


@pytest.fixture
def step_case():
    def build(
        *,
        action: ActionCode = ActionCode.STAY,
        energy: float = 20.0,
        age: int = 5,
        alive: bool = True,
        energy_changes: dict[str, float] | None = None,
        evolution_changes: dict[str, float | int] | None = None,
    ):
        raw = load_config("tests/fixtures/config/valid_v1.yaml").model_dump(mode="python")
        raw["world"].update({"width": 5, "height": 5, "resource_distribution": "uniform"})
        raw["population"].update({"initial_agents": 0, "max_agents": 4, "max_births_per_step": 2})
        raw["energy"].update(energy_changes or {})
        raw["evolution"].update(evolution_changes or {})
        config = compile_config(ExperimentConfig.model_validate(raw)).core
        rng = create_rng_state(17)
        world = initialize_world(config.world, rng.key)
        population = create_empty_population(4)
        if alive:
            population = eqx.tree_at(
                lambda p: (
                    p.alive,
                    p.agent_id,
                    p.lineage_id,
                    p.genome_id,
                    p.position,
                    p.energy,
                    p.age,
                ),
                population,
                (
                    jnp.asarray([True, False, False, False]),
                    jnp.asarray([0, -1, -1, -1], dtype=jnp.int32),
                    jnp.asarray([0, -1, -1, -1], dtype=jnp.int32),
                    jnp.asarray([0, -1, -1, -1], dtype=jnp.int32),
                    jnp.asarray([[2, 2], [-1, -1], [-1, -1], [-1, -1]], dtype=jnp.int32),
                    jnp.asarray([energy, 0, 0, 0], dtype=jnp.float32),
                    jnp.asarray([age, 0, 0, 0], dtype=jnp.int32),
                ),
            )
        world = rebuild_world_occupancy(world, population, width=5, height=5).world
        genomes = create_empty_genome_batch(4)
        if alive:
            bias = genomes.layer2.bias.at[0, int(action)].set(1.0)
            genomes = eqx.tree_at(lambda g: g.layer2.bias, genomes, bias)
        state = SimulationState(
            step=jnp.asarray(3, dtype=jnp.int32),
            rng=rng,
            ids=create_id_counters(
                next_agent_id=int(alive),
                next_genome_id=int(alive),
                next_lineage_id=int(alive),
            ),
            world=world,
            population=population,
            genomes=genomes,
        )
        return state, config

    return build
