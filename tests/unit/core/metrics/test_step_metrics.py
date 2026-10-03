# ruff: noqa: ANN001

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.metrics import StepMetrics, compute_step_metrics, create_step_event_buffer


def compute(case: dict[str, Any]) -> StepMetrics:
    events = create_step_event_buffer(
        births=case["births"], pre_action_deaths=case["pre"], post_action_deaths=case["post"]
    )
    return compute_step_metrics(
        step=jnp.asarray(2, dtype=jnp.int32),
        population=case["population"],
        genomes=case["genomes"],
        world=case["world"],
        events=events,
        feeding=case["feeding"],
        metabolism=case["metabolism"],
        action_costs=case["action_costs"],
        movement=case["movement"],
        reproduction=case["reproduction"],
    )


def test_final_state_gauges_and_step_flows(metrics_case) -> None:
    metrics = compute(metrics_case)
    assert metrics.alive_count.item() == 3
    assert metrics.total_energy_alive.item() == 18
    assert metrics.mean_energy_alive.item() == 6
    assert metrics.mean_age_alive.item() == 4
    assert metrics.total_resources.item() == 10
    assert metrics.mean_environment.item() == 0.5
    assert metrics.active_lineage_count.item() == 2
    assert metrics.birth_count.item() == 2
    assert metrics.birth_rejected_capacity.item() == 3
    assert metrics.death_count.item() == 2
    assert metrics.deaths_by_cause.tolist() == [0, 1, 1, 0, 0, 0]
    assert metrics.mean_age_at_death.item() == 6
    assert metrics.resource_consumed.item() == 3
    assert metrics.feeding_energy_gained.item() == 6
    assert metrics.basal_energy_cost.item() == 4
    assert metrics.movement_energy_cost.item() == 2
    assert metrics.feeding_energy_cost.item() == 2
    assert metrics.reproduction_energy_cost.item() == 6
    assert metrics.death_energy_removed.item() == 3
    assert metrics.movement_success_count.item() == 2
    assert metrics.feeding_success_count.item() == 2
    assert metrics.reproduction_success_count.item() == 2


def test_zero_live_or_dead_agents_produces_zero_means(metrics_case) -> None:
    metrics_case["population"] = jax.tree.map(jnp.zeros_like, metrics_case["population"])
    metrics_case["pre"] = jax.tree.map(jnp.zeros_like, metrics_case["pre"])
    metrics_case["post"] = jax.tree.map(jnp.zeros_like, metrics_case["post"])
    metrics = compute(metrics_case)
    assert metrics.mean_energy_alive.item() == 0
    assert metrics.mean_age_alive.item() == 0
    assert metrics.mean_age_at_death.item() == 0
    assert not jnp.isnan(metrics.mean_energy_alive)


def test_genetic_variance_and_eager_jit_are_equivalent(metrics_case) -> None:
    identical = compute(metrics_case)
    metrics_case["genomes"] = jax.tree.map(lambda leaf: leaf.at[1].set(1), metrics_case["genomes"])
    different = compute(metrics_case)
    compiled = eqx.filter_jit(compute)(metrics_case)
    assert identical.genetic_parameter_variance.item() == 0
    assert different.genetic_parameter_variance.item() > 0
    assert jnp.allclose(different.genetic_parameter_variance, compiled.genetic_parameter_variance)
