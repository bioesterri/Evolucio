# ruff: noqa: ANN001, ANN002, ANN201, ANN202

import hashlib
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from evolucio.persistence.local import InvalidArtifactPath, LocalArtifactRegistry, sha256_file
from evolucio.persistence.local.artifacts import atomic_write_text


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/file",
        "../file",
        "a/../../file",
        "C:/file",
        "C:file",
        "\\\\host\\share",
        "a\\..\\file",
        "",
        ".",
        "./a",
        "a//b",
        "a/",
        "a/../b",
        "manifest.json",
        "a\0b",
    ],
)
def test_path_traversal_rejected(tmp_path, path):
    registry = LocalArtifactRegistry(tmp_path)
    with pytest.raises(InvalidArtifactPath):
        registry.write_json_artifact(relative_path=path, payload={}, name="bad", kind="test")


def test_registry_record_hash_and_idempotency(tmp_path):
    registry = LocalArtifactRegistry(tmp_path)
    record = registry.write_json_artifact(
        relative_path="artifacts/a.json", payload={"é": 2}, name="a", kind="test"
    )
    path = tmp_path / record.relative_path
    assert record.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record.byte_size == path.stat().st_size
    assert record.media_type == "application/json"
    assert path.read_bytes().endswith(b"\n")
    assert (
        registry.write_json_artifact(
            relative_path=record.relative_path, payload={"é": 2}, name="a", kind="test"
        )
        is record
    )
    with pytest.raises(FrozenInstanceError):
        record.name = "changed"
    with pytest.raises(ValueError):
        registry.write_json_artifact(
            relative_path=record.relative_path, payload={"é": 3}, name="a", kind="test"
        )
    assert registry.records == (record,)
    with pytest.raises(ValueError):
        registry.write_json_artifact(relative_path="b.json", payload={}, name="a", kind="test")
    path.write_text("corruption")
    with pytest.raises(ValueError):
        registry.register_existing_artifact(
            name="a", kind="test", relative_path=record.relative_path, media_type="application/json"
        )


def test_existing_artifact_and_conflicts(tmp_path):
    (tmp_path / "existing.txt").write_text("explicitly supplied")
    registry = LocalArtifactRegistry(tmp_path)
    record = registry.register_existing_artifact(
        name="file", kind="text", relative_path="existing.txt", media_type="text/plain"
    )
    assert record.byte_size == 19
    with pytest.raises(ValueError):
        registry.register_existing_artifact(
            name="different", kind="text", relative_path="existing.txt", media_type="text/plain"
        )
    with pytest.raises(FileNotFoundError):
        registry.register_existing_artifact(
            name="missing", kind="text", relative_path="missing.txt", media_type="text/plain"
        )
    with pytest.raises(FileExistsError):
        LocalArtifactRegistry(tmp_path).write_json_artifact(
            relative_path="existing.txt", payload={}, name="file", kind="text"
        )


def test_symlink_paths_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "run"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(InvalidArtifactPath):
        LocalArtifactRegistry(root).write_json_artifact(
            relative_path="link/file.json", payload={}, name="unsafe", kind="test"
        )
    assert not list(outside.iterdir())


def test_hash_is_streaming(tmp_path):
    content = b"reproducible bytes\x00" * 150000
    path = tmp_path / "large.bin"
    path.write_bytes(content)
    with patch.object(type(path), "read_bytes", side_effect=AssertionError("must stream")):
        assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_atomic_failure_preserves_target(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    atomic_write_text(path, '{"status":"created"}\n')
    old = path.read_bytes()

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr("evolucio.persistence.local.artifacts.os.replace", fail)
    with pytest.raises(OSError):
        atomic_write_text(path, '{"status":"running"}\n')
    assert path.read_bytes() == old
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("payload", [{"bad": float("nan")}, {"bad": object()}])
def test_serialization_errors_do_not_write(tmp_path, payload):
    with pytest.raises((TypeError, ValueError)):
        LocalArtifactRegistry(tmp_path).write_json_artifact(
            relative_path="bad.json", payload=payload, name="bad", kind="test"
        )
    assert not list(tmp_path.iterdir())
