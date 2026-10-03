"""Pure invariant checks across a completed simulation step."""

# pyright: reportUnknownMemberType=false

import jax
import jax.numpy as jnp

from evolucio.core.codes import DEATH_CAUSE_COUNT, DeathCauseCode
from evolucio.core.dtypes import CODE_DTYPE, COUNT_DTYPE, MASK_DTYPE
from evolucio.core.metrics import MetricsAccumulator, StepEventBuffer, StepMetrics
from evolucio.core.rng import advance_rng
from evolucio.core.state import SimulationState
from evolucio.core.types import Array

from .codes import (
    TRANSITION_INVARIANT_CODE_COUNT,
    TransitionInvariantCode,
    TransitionInvariantReport,
)


def _scalar_count(failed: Array) -> Array:
    return failed.astype(COUNT_DTYPE)


def _report(counts: Array) -> TransitionInvariantReport:
    counts = counts.astype(COUNT_DTYPE)
    failed = (counts > 0).astype(MASK_DTYPE)
    return TransitionInvariantReport(
        violation_counts=counts,
        failed=failed,
        total_violation_count=jnp.sum(counts, dtype=COUNT_DTYPE),
        ok=(~jnp.any(failed)).astype(MASK_DTYPE),
    )


def check_step_transition_invariants(
    *,
    before: SimulationState,
    after: SimulationState,
    events: StepEventBuffer,
    metrics: StepMetrics,
    accumulator: MetricsAccumulator,
) -> TransitionInvariantReport:
    """Detect step, event, metric, RNG, and counter inconsistencies."""
    births = jnp.sum(events.births.born, dtype=COUNT_DTYPE)
    died = events.deaths.records.died
    recorded_deaths = jnp.sum(died, dtype=COUNT_DTYPE)
    deaths = events.deaths.count.astype(COUNT_DTYPE)
    alive_before = jnp.sum(before.population.alive, dtype=COUNT_DTYPE)
    alive_after = jnp.sum(after.population.alive, dtype=COUNT_DTYPE)

    causes = jnp.arange(DEATH_CAUSE_COUNT, dtype=CODE_DTYPE)
    cause_histogram = (
        jnp.sum(
            died[:, None] & (events.deaths.records.cause[:, None] == causes[None, :]),
            axis=0,
            dtype=COUNT_DTYPE,
        )
        .at[int(DeathCauseCode.NONE)]
        .set(0)
    )
    terminal_cause_count = jnp.sum(cause_histogram, dtype=COUNT_DTYPE)

    expected_rng, _ = advance_rng(before.rng)
    rng_mismatch = ~jnp.all(
        jax.random.key_data(after.rng.key) == jax.random.key_data(expected_rng.key)
    )
    regression = (
        (after.ids.next_agent_id < before.ids.next_agent_id)
        | (after.ids.next_genome_id < before.ids.next_genome_id)
        | (after.ids.next_lineage_id < before.ids.next_lineage_id)
    )
    agent_allocation_mismatch = after.ids.next_agent_id - before.ids.next_agent_id != births
    genome_allocation_mismatch = after.ids.next_genome_id - before.ids.next_genome_id != births
    parent_identity_precedes = events.births.born[None, :] & (
        events.births.parent_agent_id[None, :] < events.births.parent_agent_id[:, None]
    )
    ranks = jnp.sum(parent_identity_precedes, axis=1, dtype=COUNT_DTYPE)
    expected_agent_ids = before.ids.next_agent_id + ranks
    expected_genome_ids = before.ids.next_genome_id + ranks
    agent_allocation_mismatch = agent_allocation_mismatch | jnp.any(
        events.births.born & (events.births.child_agent_id != expected_agent_ids)
    )
    genome_allocation_mismatch = genome_allocation_mismatch | jnp.any(
        events.births.born & (events.births.child_genome_id != expected_genome_ids)
    )

    before_ids = before.population.agent_id
    after_is_new = after.population.alive & ~jnp.any(
        (after.population.agent_id[:, None] == before_ids[None, :])
        & before.population.alive[None, :],
        axis=1,
    )
    observed_new_agents = jnp.sum(after_is_new, dtype=COUNT_DTYPE)

    birth_parent_matches = (
        events.births.parent_agent_id[:, None] == before.population.agent_id[None, :]
    ) & before.population.alive[None, :]
    birth_parent_exists = jnp.any(birth_parent_matches, axis=1)
    birth_parent_slot = jnp.argmax(birth_parent_matches, axis=1)
    birth_content_mismatch = events.births.born & (
        ~after_is_new
        | (events.births.birth_step != after.population.birth_step)
        | (events.births.child_agent_id != after.population.agent_id)
        | (events.births.parent_agent_id != after.population.parent_id)
        | (events.births.lineage_id != after.population.lineage_id)
        | (events.births.generation != after.population.generation)
        | (events.births.child_genome_id != after.population.genome_id)
        | ~jnp.all(events.births.birth_position == after.population.position, axis=1)
        | (events.births.child_initial_energy != after.population.energy)
        | ~birth_parent_exists
        | (events.births.parent_genome_id != before.population.genome_id[birth_parent_slot])
    )
    birth_event_mismatches = jnp.sum(
        events.births.born != after_is_new, dtype=COUNT_DTYPE
    ) + jnp.sum(birth_content_mismatch, dtype=COUNT_DTYPE)

    removed_before = before.population.alive & ~jnp.any(
        (before.population.agent_id[:, None] == after.population.agent_id[None, :])
        & after.population.alive[None, :],
        axis=1,
    )
    death_matches_removed = jnp.any(
        died[:, None]
        & (events.deaths.records.agent_id[:, None] == before.population.agent_id[None, :])
        & removed_before[None, :],
        axis=1,
    )
    removed_has_record = jnp.any(
        died[None, :]
        & (before.population.agent_id[:, None] == events.deaths.records.agent_id[None, :]),
        axis=1,
    )
    death_event_mismatches = jnp.sum(died & ~death_matches_removed, dtype=COUNT_DTYPE) + jnp.sum(
        removed_before & ~removed_has_record, dtype=COUNT_DTYPE
    )

    counts = jnp.zeros((TRANSITION_INVARIANT_CODE_COUNT,), dtype=COUNT_DTYPE)
    counts = counts.at[TransitionInvariantCode.STEP_ADVANCE_MISMATCH].set(
        _scalar_count(after.step != before.step + 1)
    )
    counts = counts.at[TransitionInvariantCode.RNG_ADVANCE_MISMATCH].set(
        _scalar_count(rng_mismatch)
    )
    counts = counts.at[TransitionInvariantCode.ID_COUNTER_REGRESSION].set(_scalar_count(regression))
    counts = counts.at[TransitionInvariantCode.AGENT_ID_ALLOCATION_MISMATCH].set(
        _scalar_count(agent_allocation_mismatch)
    )
    counts = counts.at[TransitionInvariantCode.GENOME_ID_ALLOCATION_MISMATCH].set(
        _scalar_count(genome_allocation_mismatch)
    )
    counts = counts.at[TransitionInvariantCode.LINEAGE_COUNTER_CHANGED].set(
        _scalar_count(after.ids.next_lineage_id != before.ids.next_lineage_id)
    )
    counts = counts.at[TransitionInvariantCode.POPULATION_BALANCE_MISMATCH].set(
        _scalar_count(alive_after != alive_before + births - deaths)
    )
    counts = counts.at[TransitionInvariantCode.BIRTH_EVENT_COUNT_MISMATCH].set(
        birth_event_mismatches + _scalar_count(births != observed_new_agents)
    )
    counts = counts.at[TransitionInvariantCode.DEATH_EVENT_COUNT_MISMATCH].set(
        death_event_mismatches + _scalar_count(deaths != recorded_deaths)
    )
    counts = counts.at[TransitionInvariantCode.DEATH_CAUSE_COUNT_MISMATCH].set(
        _scalar_count(terminal_cause_count != deaths)
    )
    counts = counts.at[TransitionInvariantCode.METRIC_ALIVE_COUNT_MISMATCH].set(
        _scalar_count(metrics.alive_count != alive_after)
    )
    counts = counts.at[TransitionInvariantCode.METRIC_BIRTH_COUNT_MISMATCH].set(
        _scalar_count(metrics.birth_count != births)
    )
    counts = counts.at[TransitionInvariantCode.METRIC_DEATH_COUNT_MISMATCH].set(
        _scalar_count(metrics.death_count != deaths)
    )
    counts = counts.at[TransitionInvariantCode.METRIC_DEATH_CAUSE_MISMATCH].set(
        jnp.sum(metrics.deaths_by_cause != cause_histogram, dtype=COUNT_DTYPE)
    )
    counts = counts.at[TransitionInvariantCode.EVENT_BUFFER_OVERFLOW].set(
        _scalar_count(events.deaths.overflow)
    )
    counts = counts.at[TransitionInvariantCode.METRICS_ACCUMULATOR_OVERFLOW].set(
        _scalar_count(accumulator.count_overflow)
    )
    return _report(counts)
