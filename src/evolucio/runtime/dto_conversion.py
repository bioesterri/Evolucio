"""Host-only conversion of PR-30 telemetry; never calls device_get."""

from collections.abc import Callable

import numpy as np

from evolucio.core.codes import DEATH_CAUSE_COUNT, DeathCauseCode
from evolucio.core.evolution import BirthEventBatch
from evolucio.core.metrics import DeathEventBuffer, DeathPhaseCode, StepMetrics
from evolucio.host import (
    HOST_DTO_SCHEMA_VERSION,
    BirthEventDTO,
    ChunkTelemetryDTO,
    DeathEventDTO,
    EventBatchDTO,
    StepMetricsDTO,
)
from evolucio.host.dto import HostArray

from ._host_validation import finite, host_array, host_ndim, nonnegative
from .types import HostChunkBatch


def _temporal_steps(value: object) -> HostArray:
    steps = host_ndim(value, "metrics.step", "iu", 1)
    nonnegative(steps, "metrics.step")
    if len(steps) and any(int(step) != int(steps[0]) + i for i, step in enumerate(steps)):
        raise ValueError("metrics.step must be consecutive and ordered")
    return steps


def metrics_to_dtos(metrics: StepMetrics) -> tuple[StepMetricsDTO, ...]:
    """Convert time-stacked host metrics to exact Python scalar records."""
    steps = _temporal_steps(metrics.step)
    length = len(steps)
    columns: dict[str, HostArray] = {}
    columns["step"] = host_array(metrics.step, "step", "iu", (length,))
    columns["alive_count"] = host_array(metrics.alive_count, "alive_count", "iu", (length,))
    columns["birth_count"] = host_array(metrics.birth_count, "birth_count", "iu", (length,))
    columns["birth_rejected_capacity"] = host_array(
        metrics.birth_rejected_capacity, "birth_rejected_capacity", "iu", (length,)
    )
    columns["death_count"] = host_array(metrics.death_count, "death_count", "iu", (length,))
    columns["deaths_by_cause"] = host_array(
        metrics.deaths_by_cause, "deaths_by_cause", "iu", (length, DEATH_CAUSE_COUNT)
    )
    columns["total_energy_alive"] = host_array(
        metrics.total_energy_alive, "total_energy_alive", "f", (length,)
    )
    columns["mean_energy_alive"] = host_array(
        metrics.mean_energy_alive, "mean_energy_alive", "f", (length,)
    )
    columns["mean_age_alive"] = host_array(metrics.mean_age_alive, "mean_age_alive", "f", (length,))
    columns["mean_age_at_death"] = host_array(
        metrics.mean_age_at_death, "mean_age_at_death", "f", (length,)
    )
    columns["total_resources"] = host_array(
        metrics.total_resources, "total_resources", "f", (length,)
    )
    columns["resource_consumed"] = host_array(
        metrics.resource_consumed, "resource_consumed", "f", (length,)
    )
    columns["feeding_energy_gained"] = host_array(
        metrics.feeding_energy_gained, "feeding_energy_gained", "f", (length,)
    )
    columns["basal_energy_cost"] = host_array(
        metrics.basal_energy_cost, "basal_energy_cost", "f", (length,)
    )
    columns["movement_energy_cost"] = host_array(
        metrics.movement_energy_cost, "movement_energy_cost", "f", (length,)
    )
    columns["feeding_energy_cost"] = host_array(
        metrics.feeding_energy_cost, "feeding_energy_cost", "f", (length,)
    )
    columns["reproduction_energy_cost"] = host_array(
        metrics.reproduction_energy_cost, "reproduction_energy_cost", "f", (length,)
    )
    columns["death_energy_removed"] = host_array(
        metrics.death_energy_removed, "death_energy_removed", "f", (length,)
    )
    columns["movement_success_count"] = host_array(
        metrics.movement_success_count, "movement_success_count", "iu", (length,)
    )
    columns["feeding_success_count"] = host_array(
        metrics.feeding_success_count, "feeding_success_count", "iu", (length,)
    )
    columns["reproduction_success_count"] = host_array(
        metrics.reproduction_success_count, "reproduction_success_count", "iu", (length,)
    )
    columns["active_lineage_count"] = host_array(
        metrics.active_lineage_count, "active_lineage_count", "iu", (length,)
    )
    columns["genetic_parameter_variance"] = host_array(
        metrics.genetic_parameter_variance, "genetic_parameter_variance", "f", (length,)
    )
    columns["mean_environment"] = host_array(
        metrics.mean_environment, "mean_environment", "f", (length,)
    )
    for name, column in columns.items():
        finite(column, name)
        if column.dtype.kind in "iu":
            nonnegative(column, name)
    return tuple(
        StepMetricsDTO(
            step=int(columns["step"][i]),
            alive_count=int(columns["alive_count"][i]),
            birth_count=int(columns["birth_count"][i]),
            birth_rejected_capacity=int(columns["birth_rejected_capacity"][i]),
            death_count=int(columns["death_count"][i]),
            deaths_by_cause=tuple(int(value) for value in columns["deaths_by_cause"][i]),
            total_energy_alive=float(columns["total_energy_alive"][i]),
            mean_energy_alive=float(columns["mean_energy_alive"][i]),
            mean_age_alive=float(columns["mean_age_alive"][i]),
            mean_age_at_death=float(columns["mean_age_at_death"][i]),
            total_resources=float(columns["total_resources"][i]),
            resource_consumed=float(columns["resource_consumed"][i]),
            feeding_energy_gained=float(columns["feeding_energy_gained"][i]),
            basal_energy_cost=float(columns["basal_energy_cost"][i]),
            movement_energy_cost=float(columns["movement_energy_cost"][i]),
            feeding_energy_cost=float(columns["feeding_energy_cost"][i]),
            reproduction_energy_cost=float(columns["reproduction_energy_cost"][i]),
            death_energy_removed=float(columns["death_energy_removed"][i]),
            movement_success_count=int(columns["movement_success_count"][i]),
            feeding_success_count=int(columns["feeding_success_count"][i]),
            reproduction_success_count=int(columns["reproduction_success_count"][i]),
            active_lineage_count=int(columns["active_lineage_count"][i]),
            genetic_parameter_variance=float(columns["genetic_parameter_variance"][i]),
            mean_environment=float(columns["mean_environment"][i]),
        )
        for i in range(length)
    )


def _event_mask(value: object, name: str) -> HostArray:
    return host_ndim(value, name, "b", 2)


def _selected(
    source: object,
    name: str,
    mask: HostArray,
    kind: str,
    *,
    position: bool = False,
    nullable: bool = False,
) -> HostArray:
    shape = (*mask.shape, 2) if position else mask.shape
    column = host_array(getattr(source, name), name, kind, shape)[mask]
    finite(column, name)
    if kind == "iu":
        if nullable:
            if bool(np.any(column < -1)):
                raise ValueError(f"{name} contains invalid parent identifiers")
        else:
            nonnegative(column, name)
    return column


def birth_events_to_dtos(events: BirthEventBatch) -> tuple[BirthEventDTO, ...]:
    """Compact real births, validate selected rows and sort by biological IDs."""
    mask = _event_mask(events.born, "born")
    columns: dict[str, HostArray] = {}
    columns["birth_step"] = _selected(events, "birth_step", mask, "iu")
    columns["child_agent_id"] = _selected(events, "child_agent_id", mask, "iu")
    columns["parent_agent_id"] = _selected(events, "parent_agent_id", mask, "iu")
    columns["lineage_id"] = _selected(events, "lineage_id", mask, "iu")
    columns["generation"] = _selected(events, "generation", mask, "iu")
    columns["child_genome_id"] = _selected(events, "child_genome_id", mask, "iu")
    columns["parent_genome_id"] = _selected(events, "parent_genome_id", mask, "iu")
    columns["child_initial_energy"] = _selected(events, "child_initial_energy", mask, "f")
    columns["parent_energy_before"] = _selected(events, "parent_energy_before", mask, "f")
    columns["parent_energy_after"] = _selected(events, "parent_energy_after", mask, "f")
    columns["mutation_selected_count"] = _selected(events, "mutation_selected_count", mask, "iu")
    columns["mutation_effective_count"] = _selected(events, "mutation_effective_count", mask, "iu")
    columns["mutation_selected_weight_count"] = _selected(
        events, "mutation_selected_weight_count", mask, "iu"
    )
    columns["mutation_selected_bias_count"] = _selected(
        events, "mutation_selected_bias_count", mask, "iu"
    )
    columns["mutation_sum_abs_delta"] = _selected(events, "mutation_sum_abs_delta", mask, "f")
    columns["mutation_max_abs_delta"] = _selected(events, "mutation_max_abs_delta", mask, "f")
    position = _selected(events, "birth_position", mask, "iu", position=True)
    records = tuple(
        BirthEventDTO(
            birth_step=int(columns["birth_step"][i]),
            child_agent_id=int(columns["child_agent_id"][i]),
            parent_agent_id=int(columns["parent_agent_id"][i]),
            lineage_id=int(columns["lineage_id"][i]),
            generation=int(columns["generation"][i]),
            child_genome_id=int(columns["child_genome_id"][i]),
            parent_genome_id=int(columns["parent_genome_id"][i]),
            position_x=int(position[i, 0]),
            position_y=int(position[i, 1]),
            child_initial_energy=float(columns["child_initial_energy"][i]),
            parent_energy_before=float(columns["parent_energy_before"][i]),
            parent_energy_after=float(columns["parent_energy_after"][i]),
            mutation_selected_count=int(columns["mutation_selected_count"][i]),
            mutation_effective_count=int(columns["mutation_effective_count"][i]),
            mutation_selected_weight_count=int(columns["mutation_selected_weight_count"][i]),
            mutation_selected_bias_count=int(columns["mutation_selected_bias_count"][i]),
            mutation_sum_abs_delta=float(columns["mutation_sum_abs_delta"][i]),
            mutation_max_abs_delta=float(columns["mutation_max_abs_delta"][i]),
        )
        for i in range(int(np.count_nonzero(mask)))
    )
    keys = [(record.birth_step, record.child_agent_id) for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate birth event identity")
    return tuple(sorted(records, key=lambda record: (record.birth_step, record.child_agent_id)))


def death_events_to_dtos(events: DeathEventBuffer) -> tuple[DeathEventDTO, ...]:
    """Compact real deaths, preserving both cause and resolution phase."""
    mask = _event_mask(events.records.died, "died")
    counts = host_array(events.count, "death count", "iu", (mask.shape[0],))
    overflow = host_array(events.overflow, "death overflow", "b", (mask.shape[0],))
    if bool(np.any(overflow)) or not np.array_equal(counts, np.count_nonzero(mask, axis=1)):
        raise ValueError("death event count or overflow is inconsistent")
    columns: dict[str, HostArray] = {}
    columns["death_step"] = _selected(events.records, "death_step", mask, "iu")
    columns["agent_id"] = _selected(events.records, "agent_id", mask, "iu")
    columns["parent_id"] = _selected(events.records, "parent_id", mask, "iu", nullable=True)
    columns["lineage_id"] = _selected(events.records, "lineage_id", mask, "iu")
    columns["genome_id"] = _selected(events.records, "genome_id", mask, "iu")
    columns["generation"] = _selected(events.records, "generation", mask, "iu")
    columns["age"] = _selected(events.records, "age", mask, "iu")
    columns["energy"] = _selected(events.records, "energy", mask, "f")
    columns["cause_code"] = _selected(events.records, "cause", mask, "iu")
    phase = _selected(events, "phase", mask, "iu")
    position = _selected(events.records, "position", mask, "iu", position=True)
    if bool(
        np.any(
            (columns["cause_code"] <= int(DeathCauseCode.NONE))
            | (columns["cause_code"] >= DEATH_CAUSE_COUNT)
        )
    ):
        raise ValueError("invalid death cause code")
    if bool(
        np.any(
            (phase != int(DeathPhaseCode.PRE_ACTION)) & (phase != int(DeathPhaseCode.POST_ACTION))
        )
    ):
        raise ValueError("invalid death phase code")
    records = tuple(
        DeathEventDTO(
            death_step=int(columns["death_step"][i]),
            agent_id=int(columns["agent_id"][i]),
            parent_id=int(columns["parent_id"][i]),
            lineage_id=int(columns["lineage_id"][i]),
            genome_id=int(columns["genome_id"][i]),
            generation=int(columns["generation"][i]),
            age=int(columns["age"][i]),
            energy=float(columns["energy"][i]),
            position_x=int(position[i, 0]),
            position_y=int(position[i, 1]),
            cause_code=int(columns["cause_code"][i]),
            phase_code=int(phase[i]),
        )
        for i in range(int(np.count_nonzero(mask)))
    )
    keys = [(record.death_step, record.agent_id) for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate death event identity")
    return tuple(sorted(records, key=lambda record: (record.death_step, record.agent_id)))


def host_chunk_batch_to_dto(batch: HostChunkBatch) -> ChunkTelemetryDTO:
    """Validate and convert an already materialized chunk without device access."""
    if (
        batch.chunk_index < 0
        or batch.start_step < 0
        or batch.end_step_exclusive <= batch.start_step
    ):
        raise ValueError("invalid chunk interval or index")
    metrics = metrics_to_dtos(batch.metrics)
    if len(metrics) != batch.end_step_exclusive - batch.start_step or (
        metrics and metrics[0].step != batch.start_step
    ):
        raise ValueError("metrics do not match the chunk interval")
    births = birth_events_to_dtos(batch.events.births)
    deaths = death_events_to_dtos(batch.events.deaths)
    if any(not batch.start_step <= event.birth_step < batch.end_step_exclusive for event in births):
        raise ValueError("birth event outside the chunk interval")
    if any(not batch.start_step <= event.death_step < batch.end_step_exclusive for event in deaths):
        raise ValueError("death event outside the chunk interval")
    return ChunkTelemetryDTO(
        schema_version=HOST_DTO_SCHEMA_VERSION,
        chunk_index=batch.chunk_index,
        start_step=batch.start_step,
        end_step_exclusive=batch.end_step_exclusive,
        metrics=metrics,
        events=EventBatchDTO(births=births, deaths=deaths),
    )


class DTOChunkConsumer:
    """Adapt the unchanged PR-30 consumer contract to host DTO telemetry."""

    def __init__(self, consumer: Callable[[ChunkTelemetryDTO], None]) -> None:
        self._consumer = consumer

    def __call__(self, batch: HostChunkBatch) -> None:
        self._consumer(host_chunk_batch_to_dto(batch))
