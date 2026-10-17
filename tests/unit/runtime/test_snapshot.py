# ruff: noqa: ANN001, ANN201, ANN202

from dataclasses import fields

import equinox as eqx
import jax
import numpy as np
import pytest

from evolucio.core.metrics import create_empty_metrics_accumulator
from evolucio.core.simulation import run_chunk_impl, simulation_step
from evolucio.host import PopulationSnapshotDTO
from evolucio.runtime import simulation_state_to_snapshot


@pytest.mark.parametrize("active", [(), (2,), (0, 2, 3)])
def test_snapshot_alive_only_order_slots_values_dtypes_and_counters(snapshot_state, active):
    state = snapshot_state(active)
    snapshot = simulation_state_to_snapshot(state)
    population = snapshot.population
    ordered = sorted(active, key=lambda slot: int(state.population.agent_id[slot]))
    assert population.capacity == 5
    assert population.alive_count == len(active)
    assert population.slot_index.tolist() == ordered
    assert population.position.shape == (len(active), 2)
    for field in fields(PopulationSnapshotDTO):
        if field.name in ("capacity", "alive_count", "slot_index"):
            continue
        actual = getattr(population, field.name)
        expected = getattr(state.population, field.name)[ordered]
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        assert not actual.flags.writeable
        assert not np.shares_memory(actual, getattr(state.population, field.name))
    assert population.slot_index.dtype == np.int32
    assert not population.slot_index.flags.writeable
    assert (
        snapshot.state_step,
        snapshot.next_agent_id,
        snapshot.next_genome_id,
        snapshot.next_lineage_id,
    ) == (7, 30, 40, 50)
    assert (snapshot.world.width, snapshot.world.height) == (5, 3)
    for name in ("resources", "environment", "occupancy"):
        array = getattr(snapshot.world, name)
        np.testing.assert_array_equal(array, getattr(state.world, name))
        assert array.dtype == getattr(state.world, name).dtype
        assert not array.flags.writeable
    if active:
        with pytest.raises(ValueError):
            population.energy[0] = 0
        saved = population.energy.copy()
        state.population.energy[:] = 900
        np.testing.assert_array_equal(population.energy, saved)


def test_transfer_excludes_genomes_rng_and_only_happens_explicitly(snapshot_state, monkeypatch):
    state = snapshot_state()
    calls = []
    original = jax.device_get

    def transfer(value):
        assert len(value) == 4
        assert value[0] is state.step and value[1] is state.ids
        assert value[2] is state.population and value[3] is state.world
        calls.append(True)
        return original(value)

    monkeypatch.setattr(jax, "device_get", transfer)
    snapshot = simulation_state_to_snapshot(state)
    assert calls == [True]
    assert not hasattr(snapshot, "rng")
    assert not hasattr(snapshot, "genomes")


@pytest.mark.parametrize(
    "field,value",
    [
        ("agent_id", -1),
        ("lineage_id", -1),
        ("genome_id", -1),
        ("energy", np.nan),
        ("position", -1),
        ("position", 9),
        ("age", -1),
        ("birth_step", 8),
        ("parent_id", -2),
        ("generation", -1),
    ],
)
def test_invalid_living_data_rejected(snapshot_state, field, value):
    state = snapshot_state()
    getattr(state.population, field)[0] = value
    with pytest.raises(ValueError):
        simulation_state_to_snapshot(state)


@pytest.mark.parametrize(
    "problem",
    ["resources", "environment", "occupancy", "counter", "step", "duplicate", "shape", "genealogy"],
)
def test_invalid_snapshot_metadata_rejected(snapshot_state, problem):
    state = snapshot_state()
    if problem in ("resources", "environment"):
        getattr(state.world, problem)[0, 0] = np.inf
    elif problem == "occupancy":
        state.world.occupancy[0, 0] = -1
    elif problem == "counter":
        state.ids.next_agent_id[...] = 1
    elif problem == "step":
        state.step[...] = -1
    elif problem == "duplicate":
        state.population.agent_id[2] = state.population.agent_id[0]
    elif problem == "genealogy":
        state.population.generation[0] = 1
    else:
        state = eqx.tree_at(lambda s: s.world.environment, state, np.zeros((1, 1), np.float32))
    with pytest.raises(ValueError):
        simulation_state_to_snapshot(state)


def test_snapshot_cannot_be_used_as_simulation_state(snapshot_state):
    snapshot = simulation_state_to_snapshot(snapshot_state())
    accumulator = create_empty_metrics_accumulator()
    with pytest.raises(AttributeError):
        simulation_step(state=snapshot, accumulator=accumulator, config=None)
    with pytest.raises((AttributeError, TypeError)):
        run_chunk_impl(state=snapshot, accumulator=accumulator, config=None, chunk_steps=1)
