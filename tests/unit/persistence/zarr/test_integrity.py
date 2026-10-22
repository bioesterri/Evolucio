# ruff: noqa: ANN001, ANN201
import shutil
from unittest.mock import patch

import pytest

from evolucio.persistence.zarr.checkpoint import load_checkpoint
from evolucio.persistence.zarr.storage import (
    CheckpointIntegrityError,
    read_metadata,
    verify_store_tree,
    write_store,
)


@pytest.mark.parametrize("change", ["edit", "delete", "add", "metadata", "symlink"])
def test_corruption_is_detected_and_not_repaired(checkpoint, change, tmp_path):
    chunk = next(
        p for p in checkpoint.rglob("*") if p.is_file() and "c" in p.relative_to(checkpoint).parts
    )
    if change == "edit":
        chunk.write_bytes(b"bad chunk")
    elif change == "delete":
        chunk.unlink()
    elif change == "add":
        (chunk.parent / "9999").write_bytes(b"uncatalogued")
    elif change == "metadata":
        (checkpoint / "zarr.json").write_text("{}")
    else:
        outside = tmp_path / "outside"
        chunk.rename(outside)
        chunk.symlink_to(outside)
    with pytest.raises(CheckpointIntegrityError):
        load_checkpoint(checkpoint_path=checkpoint)
    with pytest.raises(CheckpointIntegrityError):
        verify_store_tree(checkpoint)
    if change == "edit":
        assert chunk.read_bytes() == b"bad chunk"


def test_hash_portable_and_foreign_file_excluded(checkpoint, tmp_path):
    identity = verify_store_tree(checkpoint)
    moved = tmp_path / "moved.zarr"
    shutil.copytree(checkpoint, moved)
    (moved / "notes.txt").write_text("not part of the capture contract")
    assert verify_store_tree(moved) == identity
    assert read_metadata(moved) == read_metadata(checkpoint)


def test_atomic_failure_leaves_no_final_or_temp(tmp_path):
    import numpy as np

    path = tmp_path / "step-000000001.zarr"
    with (
        patch("evolucio.persistence.zarr.storage.os.rename", side_effect=OSError("disk")),
        pytest.raises(OSError),
    ):
        write_store(path, {"value": np.asarray(1)}, {"kind": "test"}, lambda _: None)
    assert not path.exists()
    assert not list(tmp_path.iterdir())
