"""OS ownership and explicit reconciliation of unexpectedly abandoned runs."""

import subprocess
import sys
from pathlib import Path

import pytest

from evolucio.config import load_config
from evolucio.persistence.local import RunStatus, create_local_run
from evolucio.recovery.assessment import inspect_run
from evolucio.recovery.errors import RunStillActiveError
from evolucio.recovery.runner import reconcile, resume
from evolucio.runtime import RunMode, RunPlan


def test_writer_process_exit_and_reconcile(tmp_path: Path) -> None:
    recorder = create_local_run(
        config=load_config("tests/fixtures/config/valid_v1.json"),
        output_root=tmp_path,
        plan=RunPlan(4, 2),
        mode=RunMode.EXPERIMENT,
    )
    recorder.mark_running()
    script = """
import sys
import time
from pathlib import Path
from evolucio.persistence.local.ownership import run_ownership
run_dir = Path(sys.argv[1])
with run_ownership(run_dir) as status:
    assert status == 'inactive'
    (run_dir / 'artifacts/part-00000000.parquet').touch()
    (run_dir / 'artifacts/step-000000002.zarr').mkdir()
    (run_dir / 'artifacts/.datasets.json.crash.tmp').touch()
    print('ready', flush=True)
    time.sleep(30)
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(recorder.run_dir)], stdout=subprocess.PIPE, text=True
    ) as writer:
        try:
            assert writer.stdout is not None
            assert writer.stdout.readline().strip() == "ready"
            assert inspect_run(recorder.run_dir).writer_active == "active"
            with pytest.raises(RunStillActiveError):
                resume(recorder.run_dir)
            with pytest.raises(RunStillActiveError):
                reconcile(recorder.run_dir)
        finally:
            writer.kill()
            writer.wait(timeout=10)
    report = inspect_run(recorder.run_dir)
    assert report.writer_active == "inactive"
    assert report.source_status is RunStatus.RUNNING
    assert sum("unindexed artifact" in warning for warning in report.warnings) == 2
    assert any("temporary artifact" in warning for warning in report.warnings)
    assert reconcile(recorder.run_dir).source_status is RunStatus.STOPPED
    manifest = (recorder.run_dir / "manifest.json").read_bytes()
    assert reconcile(recorder.run_dir).source_status is RunStatus.STOPPED
    assert (recorder.run_dir / "manifest.json").read_bytes() == manifest
