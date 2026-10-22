# ruff: noqa: ANN001, ANN201
from dataclasses import fields

import pytest

from evolucio.config import load_config
from evolucio.host import (
    BirthEventDTO,
    ChunkTelemetryDTO,
    DeathEventDTO,
    EventBatchDTO,
    StepMetricsDTO,
)
from evolucio.persistence.local import create_local_run
from evolucio.runtime import RunMode, RunPlan


def metric(step):
    values = {f.name: 1.25 if f.type == "float" else 1 for f in fields(StepMetricsDTO)}
    values.update(step=step, deaths_by_cause=(0, 1, 2, 3, 4, 5))
    return StepMetricsDTO(**values)


def birth(step, child, parent):
    values = {f.name: 0.25 if f.type == "float" else 1 for f in fields(BirthEventDTO)}
    values.update(
        birth_step=step,
        child_agent_id=child,
        parent_agent_id=parent,
        child_genome_id=child + 100,
        parent_genome_id=parent + 100,
    )
    return BirthEventDTO(**values)


def death(step, agent):
    values = {f.name: 0.25 if f.type == "float" else 1 for f in fields(DeathEventDTO)}
    values.update(death_step=step, agent_id=agent)
    return DeathEventDTO(**values)


def chunk(index=0, start=0, end=20, births=(), deaths=()):
    return ChunkTelemetryDTO(
        1,
        index,
        start,
        end,
        tuple(metric(s) for s in range(start, end)),
        EventBatchDTO(tuple(births), tuple(deaths)),
    )


@pytest.fixture
def recorder(tmp_path):
    return create_local_run(
        config=load_config("tests/fixtures/config/valid_v1.json"),
        plan=RunPlan(20, 4),
        mode=RunMode.EXPERIMENT,
        output_root=tmp_path / "runs",
    )
