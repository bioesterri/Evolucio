# ruff: noqa: ANN001, ANN201

from dataclasses import fields

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from evolucio.config import ExperimentConfig, freeze_config, load_config
from evolucio.config.compile import build_compile_signature, compile_config
from evolucio.core.codes import ActionCode
from evolucio.core.metrics import create_empty_metrics_accumulator
from evolucio.runtime import DTOChunkConsumer, RunMode, RunPlan, run_with_mode
from evolucio.runtime.initialization import initialize_simulation
from tests.regression.helpers import assert_pytree_equivalent, canonical_case


def assert_records(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        for field in fields(a):
            left, right = getattr(a, field.name), getattr(b, field.name)
            assert left == (
                pytest.approx(right, rel=1e-6, abs=1e-6) if isinstance(right, float) else right
            )


def test_four_modes_preserve_nontrivial_trajectory():
    state, config, _ = canonical_case()
    state = eqx.tree_at(
        lambda s: s.population.energy, state, state.population.energy.at[2].set(0.26)
    )
    bias = state.genomes.layer2.bias.at[2].set(0).at[2, int(ActionCode.MOVE_EAST)].set(2)
    state = eqx.tree_at(lambda s: s.genomes.layer2.bias, state, bias)
    original = jax.tree.map(lambda value: value, config)
    accumulator = create_empty_metrics_accumulator()
    results, telemetry = {}, {}
    for mode in RunMode:
        chunks = []
        result = run_with_mode(
            mode=mode,
            state=state,
            accumulator=accumulator,
            config=config,
            plan=RunPlan(8, 3),
            consumer=DTOChunkConsumer(chunks.append),
        )
        results[mode] = result
        telemetry[mode] = chunks
        assert result.runtime.initial_step == 0
        assert result.runtime.final_step == result.runtime.steps_executed == 8
        assert eqx.tree_equal(config, original)
    reference = results[RunMode.EXPERIMENT].runtime
    for outcome in results.values():
        assert_pytree_equivalent(outcome.runtime.state, reference.state)
        assert_pytree_equivalent(outcome.runtime.accumulator, reference.accumulator)
        assert jnp.array_equal(
            jax.random.key_data(outcome.runtime.state.rng.key),
            jax.random.key_data(reference.state.rng.key),
        )
    for mode in (RunMode.SMOKE, RunMode.DEBUG):
        for name in ("metrics", "births", "deaths"):

            def records(chunks, name=name) -> list[object]:
                return [
                    record
                    for chunk in chunks
                    for record in (
                        chunk.metrics if name == "metrics" else getattr(chunk.events, name)
                    )
                ]

            assert_records(records(telemetry[mode]), records(telemetry[RunMode.EXPERIMENT]))
    assert not telemetry[RunMode.BENCHMARK]
    assert results[RunMode.DEBUG].latest_diagnostic.step == 7
    assert int(reference.accumulator.movement_success_total) > 0
    assert int(reference.accumulator.feeding_success_total) > 0
    assert int(reference.accumulator.births_total) > 0
    assert int(reference.accumulator.deaths_total) > 0
    births = [birth for chunk in telemetry[RunMode.EXPERIMENT] for birth in chunk.events.births]
    assert sum(birth.mutation_effective_count for birth in births) > 0
    benchmark = results[RunMode.BENCHMARK].benchmark
    assert benchmark.elapsed_seconds > 0 and benchmark.steps_per_second > 0
    assert benchmark.chunks_executed == 3


def test_modes_preserve_hash_signature_and_complete_empty_population():
    data = load_config("tests/fixtures/config/valid_v1.json").model_dump(mode="json")
    data["world"].update(width=3, height=3, resource_distribution="uniform")
    data["population"].update(initial_agents=0, max_agents=3, max_births_per_step=1)
    host = ExperimentConfig.model_validate(data)
    frozen, signature = freeze_config(host), build_compile_signature(host)
    core = compile_config(host).core
    state = initialize_simulation(core, seed=host.seed)
    state = eqx.tree_at(lambda s: s.step, state, state.step + 5)
    accumulator = create_empty_metrics_accumulator()
    results = [
        run_with_mode(
            mode=mode, state=state, accumulator=accumulator, config=core, plan=RunPlan(2, 1)
        )
        for mode in RunMode
    ]
    for outcome in results:
        assert outcome.runtime.initial_step == 5
        assert outcome.runtime.final_step == 7
        assert int(outcome.runtime.accumulator.steps_accumulated) == 2
        assert_pytree_equivalent(outcome.runtime.state, results[0].runtime.state)
        assert_pytree_equivalent(outcome.runtime.accumulator, results[0].runtime.accumulator)
        assert freeze_config(host).config_hash == frozen.config_hash
        assert build_compile_signature(host) == signature
        assert eqx.tree_equal(core, compile_config(host).core)
