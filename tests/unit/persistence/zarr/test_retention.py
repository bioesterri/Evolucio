# ruff: noqa: ANN001, ANN201
from unittest.mock import patch

import pytest

from evolucio.persistence.zarr.index import ZarrRunIndex, load_index
from evolucio.persistence.zarr.schemas import ZarrCapturePolicy

from .test_index import indexed_checkpoint


def test_keep_two_only_after_new_is_indexed(recorder, carry):
    writer = ZarrRunIndex(
        recorder, ZarrCapturePolicy(checkpoint_interval=10, keep_last_checkpoints=2)
    )
    for step in (10, 20):
        writer.add(indexed_checkpoint(recorder, carry, step))
    import shutil

    original = shutil.rmtree
    order = []

    def delete(path) -> None:
        order.append([r.state_step for r in load_index(recorder.run_dir).checkpoints])
        assert (recorder.run_dir / "artifacts/zarr/checkpoints/step-000000030.zarr").is_dir()
        original(path)

    third = indexed_checkpoint(recorder, carry, 30)
    with patch("evolucio.persistence.zarr.index.shutil.rmtree", delete):
        writer.add(third)
    assert order == [[10, 20, 30]]
    assert [r.state_step for r in writer.index.checkpoints] == [20, 30]
    assert not (recorder.run_dir / "artifacts/zarr/checkpoints/step-000000010.zarr").exists()
    from dataclasses import replace

    from evolucio.persistence.zarr.snapshot import write_snapshot
    from evolucio.runtime.snapshot import simulation_state_to_snapshot

    dto = simulation_state_to_snapshot(carry[0])
    for step in (10, 20, 30):
        writer.add(
            write_snapshot(
                run_dir=recorder.run_dir,
                run_id=recorder.run_id,
                snapshot=replace(dto, state_step=step),
                config_hash=recorder.manifest.config_hash,
                compile_signature=recorder.manifest.compile_signature,
            )
        )
    assert [record.state_step for record in writer.index.snapshots] == [10, 20, 30]
    writer.finalize()


def test_failed_new_store_does_not_remove_old(recorder, carry):
    writer = ZarrRunIndex(
        recorder, ZarrCapturePolicy(checkpoint_interval=10, keep_last_checkpoints=1)
    )
    first = indexed_checkpoint(recorder, carry, 10)
    writer.add(first)
    with (
        patch("evolucio.persistence.zarr.storage.os.rename", side_effect=OSError("disk")),
        pytest.raises(OSError),
    ):
        indexed_checkpoint(recorder, carry, 20)
    assert writer.index.checkpoints == [first]
    assert (recorder.run_dir / first.relative_path).exists()
    writer.finalize(complete=False)
