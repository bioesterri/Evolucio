# ruff: noqa: ANN001, ANN201

from dataclasses import replace

from evolucio.experiments import BatchPlan, SeedSet, expand_batch_plan, run_batch, seeds_for_case
from tests.experiment_helpers import write_cases


def test_paired_seeds_and_case_reordering(tmp_path):
    cases = write_cases(tmp_path)
    plan = BatchPlan(cases, SeedSet((101, 202, 303)), 4, 2)
    expected = [("A", 101), ("A", 202), ("A", 303), ("B", 101), ("B", 202), ("B", 303)]
    assert [(s.case_id, s.seed) for s in expand_batch_plan(plan)] == expected
    result = run_batch(plan=plan)
    assert [(r.case_id, r.seed) for r in result.runs] == expected
    assert seeds_for_case(result, "A") == seeds_for_case(result, "B") == (101, 202, 303)
    assert result.completed_run_count == 6
    reordered = run_batch(plan=replace(plan, cases=tuple(reversed(cases))))
    by_key = {(r.case_id, r.seed): replace(r, ordinal=1) for r in result.runs}
    assert by_key == {(r.case_id, r.seed): replace(r, ordinal=1) for r in reordered.runs}
    # The controlled ecological difference preserves the static compile structure.
    assert len({r.compile_signature for r in result.runs}) == 1
    assert result.runs[0].config_hash != result.runs[3].config_hash
