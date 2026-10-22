# ruff: noqa: ANN001, ANN201, ANN202

from dataclasses import fields

import jax
import numpy as np

from evolucio.core.metrics import create_empty_metrics_accumulator
from evolucio.core.simulation import run_chunk
from evolucio.host import ChunkTelemetryDTO
from evolucio.runtime import DTOChunkConsumer, RunPlan, run_simulation, simulation_state_to_snapshot
from tests.regression.helpers import assert_pytree_equivalent, canonical_case


def test_runtime_streams_complete_host_dtos_and_snapshot_is_explicit(monkeypatch):
    state, config, _ = canonical_case()
    accumulator = create_empty_metrics_accumulator()
    whole = run_chunk(state=state, accumulator=accumulator, config=config, chunk_steps=10)
    expected_metrics, expected_events = jax.device_get((whole.metrics, whole.events))
    batches = []
    transfers = []
    original = jax.device_get

    def transfer(value):
        transfers.append(len(value))
        assert len(value) in (2, 4)
        return original(value)

    monkeypatch.setattr(jax, "device_get", transfer)
    result = run_simulation(
        state=state,
        accumulator=accumulator,
        config=config,
        plan=RunPlan(10, 4),
        consumer=DTOChunkConsumer(batches.append),
    )
    assert transfers == [2, 2, 2]
    assert all(isinstance(batch, ChunkTelemetryDTO) for batch in batches)
    assert [(b.start_step, b.end_step_exclusive) for b in batches] == [(0, 4), (4, 8), (8, 10)]
    assert [b.chunk_index for b in batches] == [0, 1, 2]
    assert_pytree_equivalent(result.state, whole.state)
    assert_pytree_equivalent(result.accumulator, whole.accumulator)
    metrics = tuple(metric for batch in batches for metric in batch.metrics)
    assert [metric.step for metric in metrics] == list(range(10))
    for i, metric in enumerate(metrics):
        for field in fields(metric):
            expected = getattr(expected_metrics, field.name)[i]
            actual = getattr(metric, field.name)
            if field.name == "deaths_by_cause":
                assert actual == tuple(int(value) for value in expected)
            else:
                assert actual == expected.item()
    births = tuple(event for batch in batches for event in batch.events.births)
    deaths = tuple(event for batch in batches for event in batch.events.deaths)
    raw_births = expected_events.births
    raw_deaths = expected_events.deaths.records
    expected_births = sorted(
        (int(raw_births.birth_step[t, slot]), int(raw_births.child_agent_id[t, slot]))
        for t, slot in zip(*np.nonzero(raw_births.born), strict=True)
    )
    expected_deaths = sorted(
        (int(raw_deaths.death_step[t, slot]), int(raw_deaths.agent_id[t, slot]))
        for t, slot in zip(*np.nonzero(raw_deaths.died), strict=True)
    )
    assert [(event.birth_step, event.child_agent_id) for event in births] == expected_births
    assert [(event.death_step, event.agent_id) for event in deaths] == expected_deaths
    assert len(births) == int(result.accumulator.births_total) > 0
    assert len(deaths) == int(result.accumulator.deaths_total) > 0
    assert all(event.child_agent_id >= 0 for event in births)
    assert all(event.agent_id >= 0 for event in deaths)
    snapshot = simulation_state_to_snapshot(result.state)
    assert transfers == [2, 2, 2, 4]
    assert snapshot.state_step == result.final_step == 10
    assert snapshot.population.alive_count == int(result.state.population.alive.sum())
    assert snapshot.population.agent_id.tolist() == sorted(snapshot.population.agent_id.tolist())
    assert not snapshot.world.resources.flags.writeable
