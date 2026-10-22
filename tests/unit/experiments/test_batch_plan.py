# ruff: noqa: ANN001, ANN201

from dataclasses import replace
from pathlib import Path

import pytest

from evolucio.experiments import (
    BatchDefinitionError,
    BatchPlan,
    BatchResult,
    ExperimentCase,
    SeedSet,
    expand_batch_plan,
)
from evolucio.runtime import RunMode


def test_cartesian_order_and_identity(config_path, monkeypatch):
    plan = BatchPlan(
        (ExperimentCase("A", config_path), ExperimentCase("B", config_path)),
        SeedSet((11, 22, 33)),
        4,
        2,
    )
    assert plan.run_count == 6
    monkeypatch.setattr(Path, "is_file", lambda self: pytest.fail("expansion must be pure"))
    specs = expand_batch_plan(plan)
    assert isinstance(specs, tuple)
    assert [(s.case_id, s.seed) for s in specs] == [
        ("A", 11),
        ("A", 22),
        ("A", 33),
        ("B", 11),
        ("B", 22),
        ("B", 33),
    ]
    assert [s.ordinal for s in specs] == list(range(1, 7))
    assert specs[0].identity == "A/seed-11"
    assert all(s.mode == RunMode.EXPERIMENT and s.total_steps == 4 for s in specs)
    assert expand_batch_plan(plan) == specs


@pytest.mark.parametrize("case_id", ["", "   ", None])
def test_invalid_case_id(config_path, case_id):
    with pytest.raises(ValueError):
        ExperimentCase(case_id, config_path)


def test_invalid_path_type():
    with pytest.raises(TypeError):
        ExperimentCase("A", "a.json")


def test_duplicate_case_id(config_path):
    case = ExperimentCase("A", config_path)
    with pytest.raises(ValueError, match="unique"):
        BatchPlan((case, case), SeedSet((11,)), 2, 1)


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"cases": ()}, ValueError),
        ({"cases": []}, TypeError),
        ({"cases": ("bad",)}, TypeError),
        ({"seed_set": (1,)}, TypeError),
        ({"total_steps": 0}, ValueError),
        ({"chunk_steps": 0}, ValueError),
        ({"total_steps": True}, TypeError),
        ({"chunk_steps": 1.5}, TypeError),
        ({"mode": "experiment"}, TypeError),
        ({"continue_on_error": 1}, TypeError),
    ],
)
def test_plan_validation(config_path, changes, error):
    plan = BatchPlan((ExperimentCase("A", config_path),), SeedSet((11,)), 2, 1)
    with pytest.raises(error):
        replace(plan, **changes)


def test_missing_and_directory_paths(tmp_path):
    for path in (tmp_path, tmp_path / "missing.json"):
        with pytest.raises(BatchDefinitionError):
            BatchPlan((ExperimentCase("A", path),), SeedSet((11,)), 2, 1)


def test_result_rejects_inconsistent_counts():
    with pytest.raises(ValueError):
        BatchResult((), 1, 0, 0)
    with pytest.raises(TypeError):
        BatchResult([], 0, 0, 0)
