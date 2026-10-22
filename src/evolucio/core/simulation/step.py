"""Pure orchestration of one complete simulation step."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx

from evolucio.core.actions import (
    resolve_cardinal_movement,
    resolve_feeding,
    validate_and_route_actions,
)
from evolucio.core.codes import RngStreamCode
from evolucio.core.diagnostics import (
    StateInvariantReport,
    TransitionInvariantReport,
    check_final_state_invariants,
    check_step_transition_invariants,
    check_structural_invariants,
)
from evolucio.core.energy import apply_action_energy_costs, apply_basal_metabolism_and_age
from evolucio.core.evolution import (
    build_birth_events,
    mutate_newborn_genomes,
    resolve_asexual_reproduction,
)
from evolucio.core.metrics import (
    MetricsAccumulator,
    StepEventBuffer,
    StepMetrics,
    compute_step_metrics,
    create_step_event_buffer,
    update_metrics_accumulator,
)
from evolucio.core.observations import build_observations
from evolucio.core.policy import infer_policy_scores, select_actions_deterministically
from evolucio.core.rng import advance_rng, derive_stream_key
from evolucio.core.state import SimulationState
from evolucio.core.viability import (
    evaluate_reproduction_gate,
    resolve_post_action_viability,
    resolve_pre_action_viability,
)
from evolucio.core.world import update_world_for_step

if TYPE_CHECKING:
    from evolucio.config.compile import CoreConfig


class SimulationStepResult(eqx.Module):
    """Persistent carry and compact outputs from one completed step."""

    state: SimulationState
    accumulator: MetricsAccumulator
    metrics: StepMetrics
    events: StepEventBuffer


class CheckedSimulationStepResult(eqx.Module):
    """One step together with non-mutating invariant diagnostics."""

    result: SimulationStepResult
    initial_invariants: StateInvariantReport
    final_invariants: StateInvariantReport
    transition_invariants: TransitionInvariantReport


def simulation_step(
    *, state: SimulationState, accumulator: MetricsAccumulator, config: CoreConfig
) -> SimulationStepResult:
    """Execute exactly one complete ecological-evolutionary step."""
    current_step = state.step
    next_rng, step_key = advance_rng(state.rng)
    movement_key = derive_stream_key(step_key, RngStreamCode.MOVEMENT_CONFLICT)
    reproduction_key = derive_stream_key(step_key, RngStreamCode.REPRODUCTION_CONFLICT)
    placement_key = derive_stream_key(step_key, RngStreamCode.BIRTH_PLACEMENT)
    mutation_key = derive_stream_key(step_key, RngStreamCode.GENOME_MUTATION)

    world = update_world_for_step(state.world, current_step, config.world)
    metabolism_result = apply_basal_metabolism_and_age(
        state.population, basal_metabolic_cost=config.energy.basal_cost
    )
    pre_viability_result = resolve_pre_action_viability(
        population=metabolism_result.population,
        genomes=state.genomes,
        world=world,
        step=current_step,
        death_energy_threshold=config.energy.death_threshold,
        maximum_age=config.evolution.max_age,
        width=config.world.width,
        height=config.world.height,
    )
    decision_state = SimulationState(
        step=current_step,
        rng=state.rng,
        ids=state.ids,
        world=pre_viability_result.world,
        population=pre_viability_result.population,
        genomes=pre_viability_result.genomes,
    )
    observations = build_observations(decision_state, config)
    scores = infer_policy_scores(pre_viability_result.genomes, observations)
    decision_result = select_actions_deterministically(
        scores, pre_viability_result.population.alive
    )
    action_validation = validate_and_route_actions(
        proposed_actions=decision_result.proposed_actions,
        alive=pre_viability_result.population.alive,
        positions=pre_viability_result.population.position,
        resources=pre_viability_result.world.resources,
        width=config.world.width,
        height=config.world.height,
    )
    movement_result = resolve_cardinal_movement(
        population=pre_viability_result.population,
        world=pre_viability_result.world,
        action_validation=action_validation,
        movement_conflict_key=movement_key,
        width=config.world.width,
        height=config.world.height,
    )
    feeding_result = resolve_feeding(
        population=movement_result.population,
        world=movement_result.world,
        actions_after_movement=movement_result.actions_after_movement,
        maximum_energy=config.energy.max_energy,
        energy_gain_per_resource=config.energy.feeding_conversion,
        feeding_max_resource_intake=config.energy.feeding_max_resource_intake,
        width=config.world.width,
        height=config.world.height,
    )
    action_cost_result = apply_action_energy_costs(
        population=feeding_result.population,
        actions_after_feeding=feeding_result.actions_after_feeding,
        movement_codes=movement_result.movement_codes,
        feeding_codes=feeding_result.feeding_codes,
        movement_energy_cost=config.energy.movement_cost,
        feeding_energy_cost=config.energy.feeding_cost,
    )
    post_viability_result = resolve_post_action_viability(
        population=action_cost_result.population,
        genomes=pre_viability_result.genomes,
        world=feeding_result.world,
        actions_after_feeding=feeding_result.actions_after_feeding,
        step=current_step,
        death_energy_threshold=config.energy.death_threshold,
        maximum_age=config.evolution.max_age,
        width=config.world.width,
        height=config.world.height,
    )
    reproduction_gate = evaluate_reproduction_gate(
        population_before_viability=action_cost_result.population,
        population_after_viability=post_viability_result.population,
        actions_before_viability=feeding_result.actions_after_feeding,
        actions_after_viability=post_viability_result.actions_after_viability,
        reproduction_energy_threshold=config.energy.reproduction_threshold,
        minimum_reproduction_age=config.evolution.min_reproduction_age,
        reproduction_energy_cost=config.energy.reproduction_cost,
        offspring_initial_energy=config.energy.offspring_initial_energy,
        death_energy_threshold=config.energy.death_threshold,
    )
    reproduction_result = resolve_asexual_reproduction(
        population=post_viability_result.population,
        genomes=post_viability_result.genomes,
        world=post_viability_result.world,
        ids=state.ids,
        reproduction_gate=reproduction_gate,
        actions_after_viability=post_viability_result.actions_after_viability,
        step=current_step,
        reproduction_energy_cost=config.energy.reproduction_cost,
        offspring_initial_energy=config.energy.offspring_initial_energy,
        birth_placement_key=placement_key,
        reproduction_conflict_key=reproduction_key,
        death_energy_threshold=config.energy.death_threshold,
        max_births_per_step=config.population.max_births_per_step,
        width=config.world.width,
        height=config.world.height,
    )
    mutation_result = mutate_newborn_genomes(
        genomes=reproduction_result.genomes,
        population=reproduction_result.population,
        newborn_mask=reproduction_result.newborn_mask,
        genome_mutation_key=mutation_key,
        weight_mutation_rate=config.evolution.weight_mutation_rate,
        weight_mutation_sigma=config.evolution.weight_mutation_sigma,
        weight_abs_limit=config.evolution.weight_abs_limit,
        bias_mutation_rate=config.evolution.bias_mutation_rate,
        bias_mutation_sigma=config.evolution.bias_mutation_sigma,
        bias_abs_limit=config.evolution.bias_abs_limit,
    )
    birth_events = build_birth_events(
        population_before_reproduction=post_viability_result.population,
        reproduction=reproduction_result,
        mutation=mutation_result,
        step=current_step,
    )
    events = create_step_event_buffer(
        births=birth_events,
        pre_action_deaths=pre_viability_result.deaths,
        post_action_deaths=post_viability_result.deaths,
    )
    next_state = SimulationState(
        step=current_step + 1,
        rng=next_rng,
        ids=reproduction_result.ids,
        world=reproduction_result.world,
        population=reproduction_result.population,
        genomes=mutation_result.genomes,
    )
    metrics = compute_step_metrics(
        step=current_step,
        population=next_state.population,
        genomes=next_state.genomes,
        world=next_state.world,
        events=events,
        feeding=feeding_result,
        metabolism=metabolism_result,
        action_costs=action_cost_result,
        movement=movement_result,
        reproduction=reproduction_result,
    )
    next_accumulator = update_metrics_accumulator(accumulator, metrics, events)
    return SimulationStepResult(
        state=next_state, accumulator=next_accumulator, metrics=metrics, events=events
    )


def simulation_step_checked(
    *, state: SimulationState, accumulator: MetricsAccumulator, config: CoreConfig
) -> CheckedSimulationStepResult:
    """Execute the canonical step and report, but never repair, invariant failures."""
    initial = check_structural_invariants(
        state=state,
        resource_capacity=config.world.resource_capacity,
        width=config.world.width,
        height=config.world.height,
    )
    result = simulation_step(state=state, accumulator=accumulator, config=config)
    final = check_final_state_invariants(
        state=result.state,
        resource_capacity=config.world.resource_capacity,
        death_energy_threshold=config.energy.death_threshold,
        maximum_age=config.evolution.max_age,
        width=config.world.width,
        height=config.world.height,
    )
    transition = check_step_transition_invariants(
        before=state,
        after=result.state,
        events=result.events,
        metrics=result.metrics,
        accumulator=result.accumulator,
    )
    return CheckedSimulationStepResult(
        result=result,
        initial_invariants=initial,
        final_invariants=final,
        transition_invariants=transition,
    )
