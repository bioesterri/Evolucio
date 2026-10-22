"""Explicit analytical snapshots; never checkpoints or automatic chunk outputs."""

import jax
import numpy as np

from evolucio.core.state import SimulationState
from evolucio.host import (
    HOST_DTO_SCHEMA_VERSION,
    PopulationSnapshotDTO,
    SimulationSnapshotDTO,
    WorldSnapshotDTO,
)
from evolucio.host.dto import HostArray

from ._host_validation import finite, host_array, host_ndim, nonnegative


def _counter(value: object, name: str) -> int:
    array = host_array(value, name, "iu", ())
    nonnegative(array, name)
    return int(array.item())


def simulation_state_to_snapshot(state: SimulationState) -> SimulationSnapshotDTO:
    """Transfer selected state explicitly, excluding genomes, RNG and accumulator.

    This can be expensive: it transfers population and all world fields. It is
    outside JIT and the normal run fast path. state_step is the next step that
    would execute; this analytical image cannot reconstruct a resumable state.
    """
    step, ids, population, world = jax.device_get(
        (state.step, state.ids, state.population, state.world)
    )
    state_step = _counter(step, "state_step")
    next_agent_id = _counter(ids.next_agent_id, "next_agent_id")
    next_genome_id = _counter(ids.next_genome_id, "next_genome_id")
    next_lineage_id = _counter(ids.next_lineage_id, "next_lineage_id")

    resources = host_ndim(world.resources, "resources", "f", 2)
    height, width = resources.shape
    if height <= 0 or width <= 0:
        raise ValueError("world dimensions must be positive")
    resources = host_array(resources, "resources", "f", (height, width))
    environment = host_array(world.environment, "environment", "f", (height, width))
    occupancy = host_array(world.occupancy, "occupancy", "iu", (height, width))
    for name, array in (("resources", resources), ("environment", environment)):
        finite(array, name)
    nonnegative(resources, "resources")
    nonnegative(occupancy, "occupancy")

    alive = host_ndim(population.alive, "alive", "b", 1)
    capacity = len(alive)
    columns: dict[str, HostArray] = {}
    for name in (
        "agent_id",
        "parent_id",
        "lineage_id",
        "genome_id",
        "generation",
        "position",
        "energy",
        "birth_step",
        "age",
    ):
        shape = (capacity, 2) if name == "position" else (capacity,)
        array = host_array(
            getattr(population, name), name, "f" if name == "energy" else "iu", shape
        )
        selected = array[alive]
        finite(selected, name)
        if name == "parent_id":
            if bool(np.any(selected < -1)):
                raise ValueError("invalid parent_id")
        elif name != "energy":
            nonnegative(selected, name)
        columns[name] = array

    agent_ids = columns["agent_id"][alive]
    genome_ids = columns["genome_id"][alive]
    if len(np.unique(agent_ids)) != len(agent_ids) or len(np.unique(genome_ids)) != len(genome_ids):
        raise ValueError("duplicate living agent or genome IDs")
    for name, counter in (
        ("agent_id", next_agent_id),
        ("genome_id", next_genome_id),
        ("lineage_id", next_lineage_id),
    ):
        if bool(np.any(columns[name][alive] >= counter)):
            raise ValueError(f"{name} is inconsistent with its counter")
    position = columns["position"][alive]
    if bool(np.any(position[:, 0] >= width)) or bool(np.any(position[:, 1] >= height)):
        raise ValueError("living position is outside the world")
    if bool(np.any(columns["birth_step"][alive] > state_step)):
        raise ValueError("birth_step is after state_step")
    parent = columns["parent_id"][alive]
    generation = columns["generation"][alive]
    if bool(
        np.any(
            ((generation == 0) & (parent != -1))
            | ((generation > 0) & ((parent < 0) | (parent >= agent_ids)))
        )
    ):
        raise ValueError("invalid living genealogy")

    slots = np.flatnonzero(alive)
    slots = slots[np.argsort(columns["agent_id"][slots], kind="stable")]
    snapshot_population = PopulationSnapshotDTO(
        capacity=capacity,
        alive_count=len(slots),
        slot_index=np.asarray(slots, dtype=columns["agent_id"].dtype),
        agent_id=columns["agent_id"][slots],
        parent_id=columns["parent_id"][slots],
        lineage_id=columns["lineage_id"][slots],
        genome_id=columns["genome_id"][slots],
        generation=columns["generation"][slots],
        position=columns["position"][slots],
        energy=columns["energy"][slots],
        birth_step=columns["birth_step"][slots],
        age=columns["age"][slots],
    )
    return SimulationSnapshotDTO(
        schema_version=HOST_DTO_SCHEMA_VERSION,
        state_step=state_step,
        next_agent_id=next_agent_id,
        next_genome_id=next_genome_id,
        next_lineage_id=next_lineage_id,
        population=snapshot_population,
        world=WorldSnapshotDTO(
            width=width,
            height=height,
            resources=resources,
            environment=environment,
            occupancy=occupancy,
        ),
    )
