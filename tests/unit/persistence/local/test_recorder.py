# ruff: noqa: ANN001, ANN201, ANN202

import hashlib
import json
from uuid import UUID

import pytest

from evolucio.config import freeze_config, load_config, serialize_config
from evolucio.persistence.local import (
    RunErrorRecord,
    RunStatus,
    create_local_run,
    load_run_manifest,
    verify_local_run,
)
from evolucio.runtime import RunMode, RunPlan


def test_created_frozen_config_and_identity(recorder, host_config):
    assert str(UUID(recorder.run_id)) == recorder.run_id
    assert recorder.manifest.status == RunStatus.CREATED
    path = recorder.run_dir / "config.json"
    assert path.read_text() == serialize_config(host_config, format="json")
    assert load_config(path) == host_config
    assert recorder.manifest.config_hash == freeze_config(host_config).config_hash
    record = recorder.manifest.artifacts[0]
    assert record.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record.sha256 != recorder.manifest.config_hash
    assert (recorder.run_dir / "artifacts").is_dir()
    assert not (recorder.run_dir / "summary.json").exists()
    assert verify_local_run(recorder.run_dir).ok


def test_injected_uuid_and_no_overwrite(tmp_path, host_config):
    identity = "3f4677aa-1c62-4fd8-bdbf-b57f6fd35d87"
    kwargs = dict(
        config=host_config,
        plan=RunPlan(4, 2),
        mode=RunMode.EXPERIMENT,
        output_root=tmp_path,
        run_id=identity,
    )
    first = create_local_run(**kwargs)
    original = (first.run_dir / "config.json").read_bytes()
    with pytest.raises(FileExistsError):
        create_local_run(**kwargs)
    assert (first.run_dir / "config.json").read_bytes() == original
    for invalid in ("../bad", "x", identity.upper()):
        with pytest.raises(ValueError):
            create_local_run(**(kwargs | {"run_id": invalid}))


def test_completed_lifecycle_immutable_config(recorder):
    path = recorder.run_dir / "config.json"
    original = path.read_bytes()
    recorder.mark_running()
    assert recorder.manifest.started_at_utc
    recorder.mark_completed(summary={"alive_count": 2}, steps_executed=4, final_step=4)
    manifest = load_run_manifest(recorder.run_dir / "manifest.json")
    assert manifest.status == RunStatus.COMPLETED
    assert manifest.finished_at_utc and manifest.error is None
    assert manifest.final_step == manifest.steps_executed == 4
    summary = json.loads((recorder.run_dir / "summary.json").read_text())
    assert summary["run_id"] == recorder.run_id and summary["alive_count"] == 2
    assert {a.relative_path for a in manifest.artifacts} == {"config.json", "summary.json"}
    assert path.read_bytes() == original
    for action in (
        recorder.mark_running,
        lambda: recorder.mark_failed(RunErrorRecord("Error", "bad")),
        lambda: recorder.mark_completed(summary={}, steps_executed=4, final_step=4),
    ):
        with pytest.raises(ValueError):
            action()
    assert verify_local_run(recorder.run_dir).ok


@pytest.mark.parametrize("started", [False, True])
def test_failed_retains_config_and_error(recorder, started):
    if started:
        recorder.mark_running()
    error = RunErrorRecord("RuntimeError", "technical failure")
    recorder.mark_failed(error)
    manifest = load_run_manifest(recorder.run_dir / "manifest.json")
    assert manifest.status == RunStatus.FAILED
    assert manifest.error == error and manifest.finished_at_utc
    assert manifest.steps_executed is None
    assert (recorder.run_dir / "config.json").exists()
    assert (recorder.run_dir / "error.json").exists()
    assert not (recorder.run_dir / "summary.json").exists()
    assert verify_local_run(recorder.run_dir).ok
    with pytest.raises(ValueError):
        recorder.mark_running()


def test_recorder_artifact_methods(recorder):
    record = recorder.write_json_artifact(
        relative_path="artifacts/notes.json", payload={"note": "x"}, name="notes", kind="notes"
    )
    assert (
        recorder.register_artifact(
            name="notes",
            kind="notes",
            relative_path=record.relative_path,
            media_type="application/json",
        )
        == record
    )
    assert record in load_run_manifest(recorder.run_dir / "manifest.json").artifacts


def test_invalid_completion_and_double_start(recorder):
    with pytest.raises(ValueError):
        recorder.mark_completed(summary={}, steps_executed=4, final_step=4)
    recorder.mark_running()
    with pytest.raises(ValueError):
        recorder.mark_running()
    with pytest.raises(ValueError):
        recorder.mark_completed(summary={}, steps_executed=3, final_step=3)
    assert not (recorder.run_dir / "summary.json").exists()


def test_atomic_manifest_and_serialization_failure(recorder, monkeypatch):
    path = recorder.run_dir / "manifest.json"
    original = path.read_bytes()

    def fail(payload):
        raise TypeError("cannot serialize")

    with monkeypatch.context() as patch:
        patch.setattr("evolucio.persistence.local.recorder.json_text", fail)
        with pytest.raises(TypeError):
            recorder.mark_running()
    assert path.read_bytes() == original
    assert load_run_manifest(path).status == recorder.manifest.status == RunStatus.CREATED
    recorder.mark_running()
    with pytest.raises(TypeError):
        recorder.mark_completed(summary={"bad": object()}, steps_executed=4, final_step=4)
    assert load_run_manifest(path).status == RunStatus.RUNNING


def test_replace_failure_preserves_manifest_and_in_memory_lifecycle(recorder, monkeypatch):
    path = recorder.run_dir / "manifest.json"
    before = path.read_bytes()

    def fail(*args: object) -> None:
        raise OSError("interrupted before replace")

    monkeypatch.setattr("evolucio.persistence.local.artifacts.os.replace", fail)
    with pytest.raises(OSError):
        recorder.mark_running()
    assert path.read_bytes() == before
    assert load_run_manifest(path).status == recorder.manifest.status == RunStatus.CREATED
