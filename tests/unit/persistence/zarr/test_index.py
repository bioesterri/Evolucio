# ruff: noqa: ANN001, ANN201
import json
import shutil

import pytest

from evolucio.persistence.zarr.checkpoint import write_checkpoint
from evolucio.persistence.zarr.index import (
    INDEX_PATH,
    ZarrRunIndex,
    latest_checkpoint,
    list_snapshots,
    load_index,
)
from evolucio.persistence.zarr.integrity import verify_zarr_artifacts


def indexed_checkpoint(recorder, carry, step=0):
    import equinox as eqx
    import jax.numpy as jnp

    state, accumulator, config, _ = carry
    state = eqx.tree_at(lambda s: s.step, state, jnp.asarray(step, jnp.int32))
    return write_checkpoint(
        run_dir=recorder.run_dir,
        run_id=recorder.run_id,
        state=state,
        accumulator=accumulator,
        config=config,
        config_hash=recorder.manifest.config_hash,
        compile_signature=recorder.manifest.compile_signature,
    )


def test_index_zero_records(recorder):
    assert latest_checkpoint(recorder.run_dir) is None
    assert list_snapshots(recorder.run_dir) == ()
    writer = ZarrRunIndex(recorder)
    assert verify_zarr_artifacts(recorder.run_dir).ok
    writer.finalize()
    assert load_index(recorder.run_dir).complete
    with pytest.raises(ValueError):
        writer.finalize()


def test_missing_orphan_and_bad_paths(recorder, carry):
    writer = ZarrRunIndex(recorder)
    record = indexed_checkpoint(recorder, carry)
    writer.add(record)
    assert latest_checkpoint(recorder.run_dir) == record
    path = recorder.run_dir / record.relative_path
    orphan = path.with_name("step-000000099.zarr")
    shutil.copytree(path, orphan)
    assert verify_zarr_artifacts(recorder.run_dir).orphan_stores == (
        orphan.relative_to(recorder.run_dir).as_posix(),
    )
    shutil.rmtree(orphan)
    shutil.rmtree(path)
    assert verify_zarr_artifacts(recorder.run_dir).missing_stores == (record.relative_path,)
    index_path = recorder.run_dir / INDEX_PATH
    payload = json.loads(index_path.read_text())
    payload["checkpoints"][0]["relative_path"] = "../escape.zarr"
    index_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_index(recorder.run_dir)
    assert not verify_zarr_artifacts(recorder.run_dir).ok


def test_duplicate_records_rejected_before_index_rewrite(recorder, carry):
    writer = ZarrRunIndex(recorder)
    record = indexed_checkpoint(recorder, carry)
    writer.add(record)
    before = (recorder.run_dir / INDEX_PATH).read_bytes()
    with pytest.raises(ValueError, match="advance"):
        writer.add(record)
    assert (recorder.run_dir / INDEX_PATH).read_bytes() == before


def test_missing_index_schema_is_not_inferred(recorder):
    ZarrRunIndex(recorder)
    path = recorder.run_dir / INDEX_PATH
    data = json.loads(path.read_text())
    del data["schema_version"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema_version"):
        load_index(recorder.run_dir)
