"""Stable codes and fixed-shape reports for core invariant diagnostics."""

from enum import IntEnum

import equinox as eqx

from evolucio.core.types import Array


class StateInvariantCode(IntEnum):
    """Invariant violations observable in one simulation state."""

    STEP_NEGATIVE = 0
    ACTIVE_ID_INVALID = 1
    ACTIVE_AGENT_ID_DUPLICATE = 2
    ACTIVE_GENOME_ID_DUPLICATE = 3
    ACTIVE_GENEALOGY_INVALID = 4
    ACTIVE_POSITION_INVALID = 5
    ACTIVE_ENERGY_NONFINITE = 6
    ACTIVE_AGE_INVALID = 7
    ACTIVE_GENOME_NONFINITE = 8
    INACTIVE_POPULATION_NONCANONICAL = 9
    INACTIVE_GENOME_NONZERO = 10
    OCCUPANCY_NEGATIVE = 11
    OCCUPANCY_MISMATCH = 12
    RESOURCE_NONFINITE = 13
    RESOURCE_OUT_OF_RANGE = 14
    ENVIRONMENT_NONFINITE = 15
    ID_COUNTER_INVALID = 16
    FINAL_ENERGY_NOT_VIABLE = 17
    FINAL_AGE_NOT_VIABLE = 18


STATE_INVARIANT_CODE_COUNT = 19


class TransitionInvariantCode(IntEnum):
    """Invariant violations observable across one completed step."""

    STEP_ADVANCE_MISMATCH = 0
    RNG_ADVANCE_MISMATCH = 1
    ID_COUNTER_REGRESSION = 2
    AGENT_ID_ALLOCATION_MISMATCH = 3
    GENOME_ID_ALLOCATION_MISMATCH = 4
    LINEAGE_COUNTER_CHANGED = 5
    POPULATION_BALANCE_MISMATCH = 6
    BIRTH_EVENT_COUNT_MISMATCH = 7
    DEATH_EVENT_COUNT_MISMATCH = 8
    DEATH_CAUSE_COUNT_MISMATCH = 9
    METRIC_ALIVE_COUNT_MISMATCH = 10
    METRIC_BIRTH_COUNT_MISMATCH = 11
    METRIC_DEATH_COUNT_MISMATCH = 12
    METRIC_DEATH_CAUSE_MISMATCH = 13
    EVENT_BUFFER_OVERFLOW = 14
    METRICS_ACCUMULATOR_OVERFLOW = 15


TRANSITION_INVARIANT_CODE_COUNT = 16


class StateInvariantReport(eqx.Module):
    """Fixed-shape aggregate of state invariant violations."""

    violation_counts: Array
    failed: Array
    total_violation_count: Array
    ok: Array


class TransitionInvariantReport(eqx.Module):
    """Fixed-shape aggregate of transition invariant violations."""

    violation_counts: Array
    failed: Array
    total_violation_count: Array
    ok: Array
