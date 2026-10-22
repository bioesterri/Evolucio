from dataclasses import FrozenInstanceError

import pytest

from evolucio.runtime import RunPlan


def test_plan_is_immutable_and_slotted() -> None:
    plan = RunPlan(total_steps=10, chunk_steps=4)
    assert (plan.total_steps, plan.chunk_steps) == (10, 4)
    assert not hasattr(plan, "__dict__")
    with pytest.raises(FrozenInstanceError):
        plan.total_steps = 12  # type: ignore[misc]


@pytest.mark.parametrize("field", ["total_steps", "chunk_steps"])
@pytest.mark.parametrize("value", [0, -1, -10])
def test_nonpositive_values(field: str, value: int) -> None:
    arguments = {"total_steps": 10, "chunk_steps": 4, field: value}
    with pytest.raises(ValueError, match=field):
        RunPlan(**arguments)


@pytest.mark.parametrize("field", ["total_steps", "chunk_steps"])
@pytest.mark.parametrize("value", [True, False, 1.5, "4", None])
def test_non_integer_values(field: str, value: object) -> None:
    arguments = {"total_steps": 10, "chunk_steps": 4, field: value}
    with pytest.raises(TypeError, match=field):
        RunPlan(**arguments)  # type: ignore[arg-type]
