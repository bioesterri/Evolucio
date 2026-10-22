# ruff: noqa: ANN001, ANN201
from dataclasses import asdict

import pytest

from evolucio.persistence.zarr.schedule import expected_capture_steps, next_chunk_length
from evolucio.persistence.zarr.schemas import ZarrCapturePolicy


@pytest.mark.parametrize(
    "field", ["checkpoint_interval", "snapshot_interval", "keep_last_checkpoints"]
)
@pytest.mark.parametrize("value", [0, -1, True, False, 1.2])
def test_invalid_policy(field, value):
    with pytest.raises(ValueError):
        ZarrCapturePolicy(**{field: value})


def test_policy_and_final_deduplication():
    assert not ZarrCapturePolicy().enabled
    assert ZarrCapturePolicy(snapshot_on_completion=True).enabled
    assert expected_capture_steps(
        initial_step=0, total_steps=20, interval=10, on_completion=True
    ) == (10, 20)
    assert expected_capture_steps(
        initial_step=3, total_steps=8, interval=5, on_completion=True
    ) == (5, 10, 11)
    assert expected_capture_steps(initial_step=0, total_steps=2, interval=None) == ()
    with pytest.raises(ValueError):
        ZarrCapturePolicy(checkpoint_on_completion=1)
    assert set(asdict(ZarrCapturePolicy())) == {
        "checkpoint_interval",
        "snapshot_interval",
        "checkpoint_on_completion",
        "snapshot_on_completion",
        "keep_last_checkpoints",
    }


def test_exact_boundary_planning():
    policy = ZarrCapturePolicy(checkpoint_interval=1000)
    step, lengths = 0, []
    while step < 2000:
        length = next_chunk_length(
            state_step=step, remaining_steps=2000 - step, chunk_steps=256, policy=policy
        )
        lengths.append(length)
        step += length
    assert lengths == [256, 256, 256, 232] * 2
    assert (
        next_chunk_length(
            state_step=7,
            remaining_steps=8,
            chunk_steps=7,
            policy=ZarrCapturePolicy(checkpoint_interval=10, snapshot_interval=6),
        )
        == 3
    )
    with pytest.raises(ValueError):
        next_chunk_length(state_step=0, remaining_steps=0, chunk_steps=1, policy=policy)
