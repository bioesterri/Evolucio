# ruff: noqa: ANN001, ANN201

import json
from dataclasses import replace
from datetime import datetime

import pytest

from evolucio.persistence.local import (
    ManifestValidationError,
    load_run_manifest,
    manifest_to_payload,
)


def test_roundtrip_and_utc(recorder):
    manifest = recorder.manifest
    assert load_run_manifest(recorder.run_dir / "manifest.json") == manifest
    assert manifest.created_at_utc.endswith("Z")
    assert datetime.fromisoformat(manifest.created_at_utc).utcoffset().total_seconds() == 0
    assert isinstance(manifest.artifacts, tuple)
    assert manifest_to_payload(manifest)["run_id"] == recorder.run_id


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest_schema_version", 2),
        ("manifest_schema_version", True),
        ("manifest_schema_name", "future"),
        ("run_id", "../escape"),
        ("status", "stopped"),
        ("config_hash", "not-a-hash"),
        ("seed", True),
        ("seed", "42"),
        ("total_steps", 0),
        ("chunk_steps", True),
        ("mode", "invalid"),
        ("config_artifact", "../config.json"),
        ("created_at_utc", "2026-10-20T12:00:00"),
        ("steps_executed", 2),
        ("artifacts", []),
        ("unknown", 1),
    ],
)
def test_invalid_manifest_fields(recorder, field, value):
    path = recorder.run_dir / "manifest.json"
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestValidationError):
        load_run_manifest(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"relative_path": "../config.json"},
        {"sha256": "F" * 64},
        {"byte_size": True},
        {"byte_size": -1},
        {"name": ""},
    ],
)
def test_invalid_artifact_records(recorder, changes):
    path = recorder.run_dir / "manifest.json"
    payload = json.loads(path.read_text())
    payload["artifacts"][0].update(changes)
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestValidationError):
        load_run_manifest(path)


@pytest.mark.parametrize("text", ["{", "{}", '{"x":1,"x":2}', '{"seed":NaN}'])
def test_bad_json(recorder, text):
    path = recorder.run_dir / "manifest.json"
    path.write_text(text)
    with pytest.raises(ManifestValidationError):
        load_run_manifest(path)


def test_duplicate_artifacts_and_nonterminal_inconsistency(recorder):
    with pytest.raises(ValueError):
        replace(recorder.manifest, artifacts=recorder.manifest.artifacts * 2)
    with pytest.raises(ValueError):
        replace(recorder.manifest, final_step=4)
