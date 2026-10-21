# ruff: noqa: ANN001, ANN201
from dataclasses import replace

import polars as pl
import pytest

from evolucio.persistence.parquet.frames import (
    births_frame,
    deaths_frame,
    genealogy_frame,
    metrics_frame,
)
from evolucio.persistence.parquet.schemas import SCHEMAS, ParquetPersistencePolicy

from .conftest import birth, death, metric


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_invalid_interval(interval):
    with pytest.raises(ValueError):
        ParquetPersistencePolicy(metrics_interval=interval)


@pytest.mark.parametrize("flag", ["persist_births", "persist_deaths", "persist_genealogy"])
def test_events_required(flag):
    with pytest.raises(ValueError):
        ParquetPersistencePolicy(**{flag: False})


def test_schemas_and_projection():
    b = birth(3, 2**40, 1)
    for name, frame in (
        ("metrics", metrics_frame("run", [metric(3)])),
        ("births", births_frame("run", [b])),
        ("deaths", deaths_frame("run", [death(4, 2)])),
        ("genealogy", genealogy_frame("run", [b])),
    ):
        assert frame.schema == SCHEMAS[name]
    assert births_frame("run", [b])["child_agent_id"].item() == 2**40
    assert metrics_frame("run", [metric(3)])["deaths_max_age"].item() == 2
    assert SCHEMAS["metrics"]["total_energy_alive"] == pl.Float32
    assert genealogy_frame("run", [b]).equals(
        births_frame("run", [b]).select(SCHEMAS["genealogy"].names())
    )
    with pytest.raises(ValueError, match="canonical"):
        metrics_frame("run", [metric(4), metric(3)])
    with pytest.raises(ValueError, match="cause"):
        metrics_frame("run", [replace(metric(3), deaths_by_cause=())])
    with pytest.raises(ValueError, match="duplicate"):
        births_frame("run", [b, replace(b, birth_step=4)])


def test_empty_and_sampling():
    for name, builder in (
        ("births", births_frame),
        ("deaths", deaths_frame),
        ("genealogy", genealogy_frame),
        ("metrics", metrics_frame),
    ):
        assert builder("r", []).schema == SCHEMAS[name]
    assert metrics_frame(
        "r", [metric(s) for s in range(3, 12)], ParquetPersistencePolicy(metrics_interval=5)
    )["step"].to_list() == [5, 10]


def test_all_death_codes_and_phases_preserved():
    rows = [
        replace(death(step, step + 100), cause_code=cause, phase_code=phase)
        for step, (cause, phase) in enumerate((c, p) for c in range(1, 6) for p in (1, 2))
    ]
    frame = deaths_frame("r", rows)
    assert frame["cause_code"].to_list() == [row.cause_code for row in rows]
    assert frame["phase_code"].to_list() == [row.phase_code for row in rows]
    event = birth(1, 2, 1)
    from dataclasses import fields

    row = births_frame("r", [event]).row(0, named=True)
    for field in fields(event):
        assert row[field.name] == getattr(event, field.name)
