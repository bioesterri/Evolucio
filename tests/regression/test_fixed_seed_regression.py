# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import json

import jax.numpy as jnp
import pytest

from evolucio.core.metrics import METRICS_SCHEMA_DIGEST, create_empty_metrics_accumulator
from evolucio.core.simulation import (
    RUN_CHUNK_SCHEMA_DIGEST,
    SIMULATION_STEP_SCHEMA_DIGEST,
    run_chunk_impl,
)
from tests.regression.helpers import (
    FIXTURE_PATH,
    FLOAT32_ATOL,
    FLOAT32_RTOL,
    REGRESSION_SEED,
    canonical_case,
)


def close(actual: float, expected: float) -> None:
    assert actual == pytest.approx(expected, abs=FLOAT32_ATOL, rel=FLOAT32_RTOL)


def test_fixed_seed_summary_checkpoints_and_genealogy() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    state, config, config_hash = canonical_case(REGRESSION_SEED)
    result = run_chunk_impl(
        state=state,
        accumulator=create_empty_metrics_accumulator(),
        config=config,
        chunk_steps=fixture["steps"],
    )
    assert fixture["fixture_version"] == 1
    assert fixture["seed"] == REGRESSION_SEED
    assert fixture["config_hash"] == config_hash
    assert fixture["contract_digests"] == {
        "simulation_step": SIMULATION_STEP_SCHEMA_DIGEST,
        "run_chunk": RUN_CHUNK_SCHEMA_DIGEST,
        "metrics": METRICS_SCHEMA_DIGEST,
    }
    expected = fixture["expected"]
    assert result.state.step.item() == expected["final_step"]
    assert result.metrics.alive_count[-1].item() == expected["alive_count"]
    assert result.accumulator.births_total.item() == expected["births_total"]
    assert result.accumulator.deaths_total.item() == expected["deaths_total"]
    assert result.accumulator.deaths_by_cause_total.tolist() == expected["deaths_by_cause"]
    close(result.metrics.total_energy_alive[-1].item(), expected["final_total_energy"])
    close(result.metrics.total_resources[-1].item(), expected["final_total_resources"])
    assert result.metrics.active_lineage_count[-1].item() == expected["active_lineage_count"]
    assert result.state.ids.next_agent_id.item() == expected["next_agent_id"]
    assert result.state.ids.next_genome_id.item() == expected["next_genome_id"]
    assert result.state.ids.next_lineage_id.item() == expected["next_lineage_id"]

    for checkpoint in fixture["checkpoints"]:
        step = checkpoint["step"]
        for field in ("alive_count", "birth_count", "death_count", "active_lineage_count"):
            assert getattr(result.metrics, field)[step].item() == checkpoint[field]
        for field in (
            "total_energy_alive",
            "total_resources",
            "genetic_parameter_variance",
        ):
            close(getattr(result.metrics, field)[step].item(), checkpoint[field])

    actual_births = []
    for step, slot in jnp.argwhere(result.events.births.born).tolist():
        actual_births.append(
            {
                "birth_step": step,
                **{
                    field: getattr(result.events.births, field)[step, slot].item()
                    for field in (
                        "child_agent_id",
                        "parent_agent_id",
                        "lineage_id",
                        "generation",
                        "child_genome_id",
                        "parent_genome_id",
                        "mutation_selected_count",
                        "mutation_effective_count",
                    )
                },
                "mutation_sum_abs_delta": result.events.births.mutation_sum_abs_delta[
                    step, slot
                ].item(),
                "mutation_max_abs_delta": result.events.births.mutation_max_abs_delta[
                    step, slot
                ].item(),
            }
        )
    for actual, expected_birth in zip(actual_births, fixture["genealogy"], strict=True):
        for field, expected_value in expected_birth.items():
            if field.startswith("mutation_") and isinstance(expected_value, float):
                close(actual[field], expected_value)
            else:
                assert actual[field] == expected_value
