# ruff: noqa: ANN001, ANN201, ANN202

from dataclasses import fields, replace
from typing import get_type_hints

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evolucio.core.metrics import StepMetrics
from evolucio.host import StepMetricsDTO
from evolucio.runtime import (
    DTOChunkConsumer,
    birth_events_to_dtos,
    death_events_to_dtos,
    host_chunk_batch_to_dto,
    metrics_to_dtos,
)


@pytest.mark.parametrize("length", [0, 1, 4])
def test_metrics_scalar_types_fields_order_and_repeatability(host_batch, length):
    batch = host_batch(length=length)
    dtos = metrics_to_dtos(batch.metrics)
    assert len(dtos) == length
    assert dtos == metrics_to_dtos(batch.metrics)
    assert [dto.step for dto in dtos] == list(range(5, 5 + length))
    assert {field.name for field in fields(StepMetricsDTO)} == {
        field.name for field in fields(StepMetrics)
    }
    for dto in dtos:
        for field in fields(dto):
            value = getattr(dto, field.name)
            expected = get_type_hints(StepMetricsDTO)[field.name]
            if field.name == "deaths_by_cause":
                assert type(value) is tuple and len(value) == 6
                assert all(type(item) is int for item in value)
            else:
                assert type(value) is expected
            assert not isinstance(value, (jax.Array, np.ndarray))


@pytest.mark.parametrize("steps", [[5, 7, 8], [5, 5, 6], [6, 5, 4]])
def test_bad_metric_order_is_rejected(host_batch, steps):
    batch = host_batch()
    batch.metrics.step[:] = steps
    with pytest.raises(ValueError, match="consecutive"):
        metrics_to_dtos(batch.metrics)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_metrics_rejected(host_batch, value):
    batch = host_batch()
    batch.metrics.mean_energy_alive[0] = value
    with pytest.raises(ValueError, match="non-finite"):
        metrics_to_dtos(batch.metrics)


def test_metric_shape_dtype_and_device_arrays_rejected(host_batch):
    batch = host_batch()
    for replacement in (np.zeros((3, 5), np.int32), np.zeros((3, 6), np.float32)):
        broken = eqx.tree_at(lambda m: m.deaths_by_cause, batch.metrics, replacement)
        with pytest.raises(ValueError, match="shape or dtype"):
            metrics_to_dtos(broken)
    broken = eqx.tree_at(lambda m: m.step, batch.metrics, jnp.arange(5, 8))
    with pytest.raises(ValueError, match="NumPy"):
        metrics_to_dtos(broken)


def test_empty_events_drop_inactive_sentinels_and_nonfinite_rows(host_batch):
    batch = host_batch()
    batch.events.births.child_initial_energy[:] = np.nan
    batch.events.deaths.records.energy[:] = np.nan
    dto = host_chunk_batch_to_dto(batch)
    assert dto.events.births == dto.events.deaths == ()


def test_events_canonical_order_and_all_fields_preserved(host_batch, add_birth, add_death):
    batch = host_batch()
    add_birth(batch, t=1, slot=0, child=12)
    add_birth(batch, t=0, slot=3, child=11)
    add_birth(batch, t=1, slot=2, child=10)
    add_death(batch, t=2, slot=0, agent=8, phase=2)
    add_death(batch, t=0, slot=3, agent=9, phase=1)
    add_death(batch, t=2, slot=2, agent=7, phase=1)
    dto = host_chunk_batch_to_dto(batch)
    assert [(b.birth_step, b.child_agent_id) for b in dto.events.births] == [
        (5, 11),
        (6, 10),
        (6, 12),
    ]
    assert [(d.death_step, d.agent_id) for d in dto.events.deaths] == [(5, 9), (7, 7), (7, 8)]
    birth = dto.events.births[0]
    assert (birth.parent_agent_id, birth.lineage_id, birth.generation) == (1, 0, 1)
    assert (birth.child_genome_id, birth.parent_genome_id) == (111, 101)
    assert (birth.position_x, birth.position_y, birth.child_initial_energy) == (2, 3, 10.0)
    assert (birth.parent_energy_before, birth.parent_energy_after) == (60.0, 30.0)
    assert (
        birth.mutation_selected_count,
        birth.mutation_effective_count,
        birth.mutation_selected_weight_count,
        birth.mutation_selected_bias_count,
    ) == (3, 1, 2, 1)
    assert (birth.mutation_sum_abs_delta, birth.mutation_max_abs_delta) == (0.125, 0.125)
    death = dto.events.deaths[-1]
    assert (death.phase_code, death.cause_code, death.parent_id) == (2, 1, -1)
    assert (death.position_x, death.position_y, death.age, death.energy) == (1, 2, 5, -0.25)
    assert dto.metrics[2].deaths_by_cause == (0, 2, 0, 0, 0, 0)
    assert dto == host_chunk_batch_to_dto(batch)
    assert "slot" not in {field.name for field in fields(birth)}
    assert "born" not in {field.name for field in fields(birth)}
    assert "died" not in {field.name for field in fields(death)}


@pytest.mark.parametrize(
    "kind,name,value",
    [
        ("birth", "child_agent_id", -1),
        ("birth", "parent_agent_id", -1),
        ("birth", "child_genome_id", -1),
        ("birth", "child_initial_energy", np.nan),
        ("birth", "mutation_max_abs_delta", np.inf),
        ("birth", "birth_position", -1),
        ("death", "agent_id", -1),
        ("death", "parent_id", -2),
        ("death", "energy", np.nan),
        ("death", "position", -1),
        ("death", "cause", 0),
    ],
)
def test_corrupt_real_events_rejected(host_batch, add_birth, add_death, kind, name, value):
    batch = host_batch()
    if kind == "birth":
        add_birth(batch)
        getattr(batch.events.births, name)[0, 0] = value

        def operation():
            return birth_events_to_dtos(batch.events.births)
    else:
        add_death(batch)
        getattr(batch.events.deaths.records, name)[0, 0] = value

        def operation():
            return death_events_to_dtos(batch.events.deaths)

    with pytest.raises(ValueError):
        operation()


@pytest.mark.parametrize("problem", ["count", "overflow", "phase", "duplicate"])
def test_invalid_death_metadata_rejected(host_batch, add_death, problem):
    batch = host_batch()
    add_death(batch)
    if problem == "count":
        batch.events.deaths.count[0] = 0
    elif problem == "overflow":
        batch.events.deaths.overflow[0] = True
    elif problem == "phase":
        batch.events.deaths.phase[0, 0] = 0
    else:
        add_death(batch, slot=1)
    with pytest.raises(ValueError):
        death_events_to_dtos(batch.events.deaths)


def test_duplicate_birth_rejected(host_batch, add_birth):
    batch = host_batch()
    add_birth(batch, slot=0)
    add_birth(batch, slot=1)
    with pytest.raises(ValueError, match="duplicate"):
        birth_events_to_dtos(batch.events.births)


@pytest.mark.parametrize("problem", ["length", "start", "birth", "death", "index", "empty"])
def test_chunk_interval_validation(host_batch, add_birth, add_death, problem):
    batch = host_batch()
    if problem == "length":
        batch = replace(batch, end_step_exclusive=9)
    elif problem == "start":
        batch = replace(batch, start_step=4, end_step_exclusive=7)
    elif problem == "birth":
        add_birth(batch)
        batch.events.births.birth_step[0, 0] = 8
    elif problem == "death":
        add_death(batch)
        batch.events.deaths.records.death_step[0, 0] = 4
    elif problem == "index":
        batch = replace(batch, chunk_index=-1)
    else:
        batch = replace(batch, end_step_exclusive=5)
    with pytest.raises(ValueError):
        host_chunk_batch_to_dto(batch)


def test_adapter_no_device_access_and_exception_propagation(host_batch, monkeypatch):
    def fail(*args: object, **kwargs: object) -> None:
        pytest.fail("conversion must not call device_get")

    monkeypatch.setattr(jax, "device_get", fail)
    batch = host_batch()
    collected = []
    DTOChunkConsumer(collected.append)(batch)
    assert collected == [host_chunk_batch_to_dto(batch)]
    error = RuntimeError("consumer failed")

    def consumer(dto):
        raise error

    with pytest.raises(RuntimeError) as caught:
        DTOChunkConsumer(consumer)(batch)
    assert caught.value is error


def test_slot_permutation_does_not_change_external_event_order(host_batch, add_birth, add_death):
    batch = host_batch()
    add_birth(batch, slot=0, child=20)
    add_birth(batch, slot=3, child=10)
    add_death(batch, slot=0, agent=9)
    add_death(batch, slot=2, agent=3, phase=2)
    events = jax.tree.map(lambda array: array[:, ::-1] if array.ndim >= 2 else array, batch.events)
    permuted = replace(batch, events=events)
    assert host_chunk_batch_to_dto(batch) == host_chunk_batch_to_dto(permuted)
