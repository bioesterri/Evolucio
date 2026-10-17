"""Device-independent immutable data contracts; snapshots are not checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

HostArray = NDArray[Any]


def _own_arrays(snapshot: PopulationSnapshotDTO | WorldSnapshotDTO) -> None:
    """Give every snapshot array its own read-only, dtype-preserving storage."""
    for field in fields(snapshot):
        if field.name in ("capacity", "alive_count", "width", "height"):
            continue
        value = getattr(snapshot, field.name)
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{field.name} must be a NumPy host array")
        owned = np.array(cast(HostArray, value), copy=True)
        owned.setflags(write=False)
        object.__setattr__(snapshot, field.name, owned)


@dataclass(frozen=True, slots=True)
class StepMetricsDTO:
    """One compact host record with Python scalar values."""

    step: int
    alive_count: int
    birth_count: int
    birth_rejected_capacity: int
    death_count: int
    deaths_by_cause: tuple[int, ...]
    total_energy_alive: float
    mean_energy_alive: float
    mean_age_alive: float
    mean_age_at_death: float
    total_resources: float
    resource_consumed: float
    feeding_energy_gained: float
    basal_energy_cost: float
    movement_energy_cost: float
    feeding_energy_cost: float
    reproduction_energy_cost: float
    death_energy_removed: float
    movement_success_count: int
    feeding_success_count: int
    reproduction_success_count: int
    active_lineage_count: int
    genetic_parameter_variance: float
    mean_environment: float


@dataclass(frozen=True, slots=True)
class BirthEventDTO:
    """One compact host record with Python scalar values."""

    birth_step: int
    child_agent_id: int
    parent_agent_id: int
    lineage_id: int
    generation: int
    child_genome_id: int
    parent_genome_id: int
    position_x: int
    position_y: int
    child_initial_energy: float
    parent_energy_before: float
    parent_energy_after: float
    mutation_selected_count: int
    mutation_effective_count: int
    mutation_selected_weight_count: int
    mutation_selected_bias_count: int
    mutation_sum_abs_delta: float
    mutation_max_abs_delta: float


@dataclass(frozen=True, slots=True)
class DeathEventDTO:
    """One compact host record with Python scalar values."""

    death_step: int
    agent_id: int
    parent_id: int
    lineage_id: int
    genome_id: int
    generation: int
    age: int
    energy: float
    position_x: int
    position_y: int
    cause_code: int
    phase_code: int


@dataclass(frozen=True, slots=True)
class EventBatchDTO:
    """Real events in canonical (step, biological identifier) order."""

    births: tuple[BirthEventDTO, ...]
    deaths: tuple[DeathEventDTO, ...]


@dataclass(frozen=True, slots=True)
class ChunkTelemetryDTO:
    """Complete telemetry over [start_step, end_step_exclusive)."""

    schema_version: int
    chunk_index: int
    start_step: int
    end_step_exclusive: int
    metrics: tuple[StepMetricsDTO, ...]
    events: EventBatchDTO


@dataclass(frozen=True, slots=True, eq=False)
class PopulationSnapshotDTO:
    """Living agents ordered by agent_id; slot_index is diagnostic, not identity."""

    capacity: int
    alive_count: int
    slot_index: HostArray
    agent_id: HostArray
    parent_id: HostArray
    lineage_id: HostArray
    genome_id: HostArray
    generation: HostArray
    position: HostArray
    energy: HostArray
    birth_step: HostArray
    age: HostArray

    def __post_init__(self) -> None:
        _own_arrays(self)


@dataclass(frozen=True, slots=True, eq=False)
class WorldSnapshotDTO:
    """Owned read-only host world fields with unchanged NumPy dtypes."""

    width: int
    height: int
    resources: HostArray
    environment: HostArray
    occupancy: HostArray

    def __post_init__(self) -> None:
        _own_arrays(self)


@dataclass(frozen=True, slots=True)
class SimulationSnapshotDTO:
    """Analytical image; state_step is the next step, with no resume guarantee."""

    schema_version: int
    state_step: int
    next_agent_id: int
    next_genome_id: int
    next_lineage_id: int
    population: PopulationSnapshotDTO
    world: WorldSnapshotDTO
