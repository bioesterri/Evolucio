# ruff: noqa: ANN001, ANN201

import json
from unittest.mock import Mock

from evolucio.cli.main import main
from evolucio.persistence.local import RunStatus, load_run_manifest, verify_local_run


def test_real_cli_and_verify_inspect(config_path, tmp_path, capsys):
    root = tmp_path / "records"
    args = [
        "run",
        str(config_path),
        "--steps",
        "4",
        "--chunk-steps",
        "2",
        "--output-root",
        str(root),
    ]
    assert main([*args, "--dry-run"]) == 0
    assert not root.exists()
    capsys.readouterr()
    assert main(args) == 0
    output = capsys.readouterr()
    (directory,) = root.iterdir()
    assert directory.name in output.out and "manifest:" in output.out
    assert verify_local_run(directory).ok
    assert main(["artifacts", "inspect", str(directory)]) == 0
    assert directory.name in capsys.readouterr().out
    assert main(["artifacts", "verify", str(directory)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    (directory / "config.json").write_text("corrupt")
    assert main(["artifacts", "verify", str(directory)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert main(["artifacts", "verify", str(tmp_path / "missing")]) == 2
    assert capsys.readouterr().err
    (directory / "manifest.json").write_text("{}")
    assert main(["artifacts", "verify", str(directory)]) == 2
    assert capsys.readouterr().err


def test_cli_initialization_failure_has_manifest(config_path, tmp_path, monkeypatch, capsys):
    root = tmp_path / "records"
    monkeypatch.setattr(
        "evolucio.runtime.initialization.initialize_simulation",
        Mock(side_effect=RuntimeError("controlled")),
    )
    assert main(["run", str(config_path), "--steps", "2", "--output-root", str(root)]) == 1
    output = capsys.readouterr()
    (directory,) = root.iterdir()
    assert directory.name in output.out and directory.name in output.err
    assert load_run_manifest(directory / "manifest.json").status == RunStatus.FAILED
    assert verify_local_run(directory).ok


def test_cli_invalid_config_creates_no_run(config_path, tmp_path, capsys):
    root = tmp_path / "records"
    config_path.write_text("{}")
    assert main(["run", str(config_path), "--steps", "2", "--output-root", str(root)]) == 2
    assert not root.exists()
    assert capsys.readouterr().err


def test_dry_runs_never_allocate_uuid(config_path, tmp_path, monkeypatch, capsys):
    generate = Mock(side_effect=AssertionError("dry-run cannot allocate UUID"))
    monkeypatch.setattr("evolucio.persistence.local.recorder.uuid4", generate)
    root = tmp_path / "not-created"
    assert (
        main(["run", str(config_path), "--steps", "2", "--output-root", str(root), "--dry-run"])
        == 0
    )
    assert (
        main(
            [
                "batch",
                "run",
                "--config",
                f"A={config_path}",
                "--seeds",
                "11,22",
                "--steps",
                "2",
                "--chunk-steps",
                "1",
                "--output-root",
                str(root),
                "--dry-run",
            ]
        )
        == 0
    )
    generate.assert_not_called()
    assert not root.exists()
    assert capsys.readouterr().out
