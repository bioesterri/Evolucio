# ruff: noqa: ANN001, ANN201, ANN202

from dataclasses import fields
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest

from evolucio.core.evolution import BirthEventBatch
from evolucio.core.ids import IdCounters
from evolucio.core.metrics import DeathEventBuffer, StepEventBuffer, StepMetrics
from evolucio.core.state import PopulationState, SimulationState, WorldState
from evolucio.core.viability import DeathRecordBatch
from evolucio.host import StepMetricsDTO
from evolucio.runtime import HostChunkBatch


@pytest.fixture
def host_batch():
    def build(length=3, capacity=4, start=5):
        metric_values = {
            field.name: np.zeros((length,), dtype=np.int32 if typ is int else np.float32)
            for field in fields(StepMetricsDTO)
            for typ in [get_type_hints(StepMetricsDTO)[field.name]]
        }
        metric_values["step"] = np.arange(start, start + length, dtype=np.int32)
        metric_values["deaths_by_cause"] = np.zeros((length, 6), dtype=np.int32)
        float_births = {
            "child_initial_energy",
            "parent_energy_before",
            "parent_energy_after",
            "mutation_sum_abs_delta",
            "mutation_max_abs_delta",
        }
        births = {}
        for field in fields(BirthEventBatch):
            shape = (length, capacity, 2) if field.name == "birth_position" else (length, capacity)
            dtype = (
                np.bool_
                if field.name == "born"
                else np.float32
                if field.name in float_births
                else np.int32
            )
            births[field.name] = np.zeros(shape, dtype=dtype)
            if "id" in field.name or field.name == "birth_position":
                births[field.name].fill(-1)
        records = {}
        for field in fields(DeathRecordBatch):
            shape = (length, capacity, 2) if field.name == "position" else (length, capacity)
            dtype = (
                np.bool_
                if field.name == "died"
                else np.float32
                if field.name == "energy"
                else np.int32
            )
            records[field.name] = np.zeros(shape, dtype=dtype)
            if "id" in field.name or field.name == "position":
                records[field.name].fill(-1)
        return HostChunkBatch(
            chunk_index=0,
            start_step=start,
            end_step_exclusive=start + length,
            metrics=StepMetrics(**metric_values),
            events=StepEventBuffer(
                births=BirthEventBatch(**births),
                deaths=DeathEventBuffer(
                    records=DeathRecordBatch(**records),
                    phase=np.zeros((length, capacity), np.int8),
                    count=np.zeros(length, np.int32),
                    overflow=np.zeros(length, np.bool_),
                ),
            ),
        )

    return build


@pytest.fixture
def add_birth():
    def add(batch, t=0, slot=0, child=10, parent=1):
        events = batch.events.births
        events.born[t, slot] = True
        for name, value in {
            "birth_step": batch.start_step + t,
            "child_agent_id": child,
            "parent_agent_id": parent,
            "lineage_id": 0,
            "generation": 1,
            "child_genome_id": child + 100,
            "parent_genome_id": parent + 100,
            "child_initial_energy": 10.0,
            "parent_energy_before": 60.0,
            "parent_energy_after": 30.0,
            "mutation_selected_count": 3,
            "mutation_effective_count": 1,
            "mutation_selected_weight_count": 2,
            "mutation_selected_bias_count": 1,
            "mutation_sum_abs_delta": 0.125,
            "mutation_max_abs_delta": 0.125,
        }.items():
            getattr(events, name)[t, slot] = value
        events.birth_position[t, slot] = [2, 3]
        batch.metrics.birth_count[t] += 1

    return add


@pytest.fixture
def add_death():
    def add(batch, t=0, slot=0, agent=3, phase=1):
        events = batch.events.deaths
        events.records.died[t, slot] = True
        for name, value in {
            "death_step": batch.start_step + t,
            "agent_id": agent,
            "parent_id": -1,
            "lineage_id": agent,
            "genome_id": agent + 100,
            "generation": 0,
            "age": 5,
            "energy": -0.25,
            "cause": 1,
        }.items():
            getattr(events.records, name)[t, slot] = value
        events.records.position[t, slot] = [1, 2]
        events.phase[t, slot] = phase
        events.count[t] += 1
        batch.metrics.death_count[t] += 1
        batch.metrics.deaths_by_cause[t, 1] += 1

    return add


@pytest.fixture
def snapshot_state():
    def build(active=(0, 2, 3)):
        capacity = 5
        values = {}
        for field in fields(PopulationState):
            shape = (capacity, 2) if field.name == "position" else (capacity,)
            dtype = (
                np.bool_
                if field.name == "alive"
                else np.float32
                if field.name == "energy"
                else np.int32
            )
            values[field.name] = np.zeros(shape, dtype=dtype)
            if "id" in field.name or field.name == "position":
                values[field.name].fill(-1)
        for slot in active:
            values["alive"][slot] = True
            for name in ("agent_id", "lineage_id", "genome_id"):
                values[name][slot] = 20 - slot
            values["energy"][slot] = 10 + slot
            values["age"][slot] = 2 + slot
            values["position"][slot] = [slot, slot % 3]
            values["birth_step"][slot] = 1
        population = PopulationState(**values)
        world = WorldState(
            resources=np.arange(15, dtype=np.float32).reshape(3, 5),
            environment=np.full((3, 5), 0.25, dtype=np.float32),
            occupancy=np.zeros((3, 5), dtype=np.int32),
        )
        for slot in active:
            x, y = values["position"][slot]
            world.occupancy[y, x] += 1
        ids = IdCounters(
            next_agent_id=np.asarray(30, np.int32),
            next_genome_id=np.asarray(40, np.int32),
            next_lineage_id=np.asarray(50, np.int32),
        )
        return SimulationState(
            step=np.asarray(7, np.int32),
            ids=ids,
            population=population,
            world=world,
            rng=SimpleNamespace(secret="excluded"),
            genomes=SimpleNamespace(secret="excluded"),
        )

    return build
