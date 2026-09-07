import jax.numpy as jnp

from evolucio.core.state import PopulationState


def population(
    energy: list[float], alive: list[bool], age: list[int] | None = None
) -> PopulationState:
    capacity = len(energy)
    zeros = jnp.zeros((capacity,), dtype=jnp.int32)
    return PopulationState(
        alive=jnp.asarray(alive, dtype=jnp.bool_),
        agent_id=jnp.arange(capacity, dtype=jnp.int32),
        parent_id=zeros,
        lineage_id=zeros,
        genome_id=zeros,
        generation=zeros,
        position=jnp.zeros((capacity, 2), dtype=jnp.int32),
        energy=jnp.asarray(energy, dtype=jnp.float32),
        birth_step=zeros,
        age=zeros if age is None else jnp.asarray(age, dtype=jnp.int32),
    )
