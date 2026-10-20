# ruff: noqa: ANN001, ANN201

import json

import pytest

from evolucio.persistence.local import verify_local_run


def test_tampered_config_hash_mismatch(recorder):
    path = recorder.run_dir / "config.json"
    path.write_bytes(path.read_bytes() + b" ")
    report = verify_local_run(recorder.run_dir)
    assert not report.ok and report.hash_mismatches == ("config.json",)


@pytest.mark.parametrize("deleted", [True, False])
def test_missing_or_modified_artifact(recorder, deleted):
    recorder.write_json_artifact(
        relative_path="artifacts/extra.json", payload={"x": 1}, name="extra", kind="test"
    )
    path = recorder.run_dir / "artifacts/extra.json"
    if deleted:
        path.unlink()
    else:
        path.write_text('{"x":2}\n')
    report = verify_local_run(recorder.run_dir)
    assert not report.ok
    assert (report.missing_artifacts if deleted else report.hash_mismatches) == (
        "artifacts/extra.json",
    )


def test_symlink_escape_and_run_directory_name(recorder, tmp_path):
    path = recorder.run_dir / "config.json"
    target = tmp_path / "external.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    report = verify_local_run(recorder.run_dir)
    assert not report.ok and report.invalid_paths == ("config.json",)
    path.unlink()
    path.write_bytes(target.read_bytes())
    moved = recorder.run_dir.with_name("wrong-id")
    recorder.run_dir.rename(moved)
    assert verify_local_run(moved).invalid_paths == ("run_id",)


def test_semantic_hash_verified_too(recorder):
    path = recorder.run_dir / "manifest.json"
    payload = json.loads(path.read_text())
    payload["config_hash"] = "0" * 64
    path.write_text(json.dumps(payload))
    assert verify_local_run(recorder.run_dir).hash_mismatches == ("config.json",)
