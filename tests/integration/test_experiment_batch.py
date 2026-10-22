# ruff: noqa: ANN001, ANN003, ANN201, ANN202

from dataclasses import replace

import jax
import numpy as np
import pytest

from evolucio.experiments import BatchPlan, BatchRunStatus, SeedSet, run_batch
from evolucio.runtime import RunMode
from tests.experiment_helpers import write_cases
from tests.regression.helpers import assert_pytree_equivalent


def test_independent_seed_twenty_reproducible_and_fresh(tmp_path, monkeypatch):
    from evolucio.experiments import batch
    from evolucio.runtime import initialization

    original_run = batch.run_with_mode
    original_init = initialization.initialize_simulation
    initial_states, initial_accumulators, final_results = [], [], []

    def initialize(config, *, seed):
        state = original_init(config, seed=seed)
        initial_states.append(state)
        assert int(state.step) == 0
        assert int(state.ids.next_agent_id) == 2
        assert int(state.ids.next_genome_id) == 2
        assert int(state.ids.next_lineage_id) == 2
        np.testing.assert_array_equal(
            jax.random.key_data(state.rng.key),
            jax.random.key_data(jax.random.key(seed, impl="threefry2x32")),
        )
        return state

    def execute(**kwargs):
        accumulator = kwargs["accumulator"]
        initial_accumulators.append(accumulator)
        assert int(accumulator.steps_accumulated) == 0
        assert int(accumulator.births_total) == int(accumulator.deaths_total) == 0
        outcome = original_run(**kwargs)
        final_results.append(outcome.runtime)
        return outcome

    monkeypatch.setattr(initialization, "initialize_simulation", initialize)
    monkeypatch.setattr(batch, "run_with_mode", execute)
    case = write_cases(tmp_path)[0]
    plan = BatchPlan((case,), SeedSet((10, 20)), 6, 2)
    together = run_batch(plan=plan)
    alone = run_batch(plan=replace(plan, seed_set=SeedSet((20,))))
    repeated = run_batch(plan=plan)
    assert together == repeated
    assert replace(together.runs[1], ordinal=1) == alone.runs[0]
    assert len({id(s) for s in initial_states}) == 5
    assert len({id(s.rng) for s in initial_states}) == 5
    assert len({id(s.ids) for s in initial_states}) == 5
    assert len({id(a) for a in initial_accumulators}) == 5
    paired, single = final_results[1], final_results[2]
    for attr in ("population", "world", "genomes", "ids"):
        assert_pytree_equivalent(getattr(paired.state, attr), getattr(single.state, attr))
    assert_pytree_equivalent(paired.accumulator, single.accumulator)
    np.testing.assert_array_equal(
        jax.random.key_data(paired.state.rng.key), jax.random.key_data(single.state.rng.key)
    )
    assert together.runs[1].births_total > 0
    assert together.runs[1].deaths_total > 0


@pytest.mark.parametrize("mode", list(RunMode))
def test_modes_and_per_run_consumers(tmp_path, mode):
    cases = write_cases(tmp_path)
    records, received_specs = {}, []

    def factory(spec):
        received_specs.append(spec)
        records[spec.identity] = []
        return records[spec.identity].append

    plan = BatchPlan(cases, SeedSet((11, 22)), 2, 1, mode)
    result = run_batch(plan=plan, consumer_factory=factory)
    assert result.completed_run_count == result.requested_run_count == 4
    assert all(r.mode == mode and r.final_step == 2 for r in result.runs)
    assert [(s.case_id, s.seed) for s in received_specs] == [
        ("A", 11),
        ("A", 22),
        ("B", 11),
        ("B", 22),
    ]
    assert all(
        len(chunks) == (0 if mode == RunMode.BENCHMARK else 2) for chunks in records.values()
    )


def test_consumer_failure_is_a_run_failure_and_next_run_is_fresh(tmp_path):
    case = write_cases(tmp_path)[0]

    def factory(spec):
        if spec.seed == 11:

            def fail(chunk):
                raise OSError("sink failed")

            return fail
        return None

    plan = BatchPlan((case,), SeedSet((11, 22)), 2, 1)
    result = run_batch(plan=plan, consumer_factory=factory)
    assert result.runs[0].status == BatchRunStatus.FAILED
    assert result.runs[0].error_type == "OSError"
    single = run_batch(plan=replace(plan, seed_set=SeedSet((22,))))
    assert replace(result.runs[1], ordinal=1) == single.runs[0]
