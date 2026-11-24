"""Project small, canonical files into the MLflow artifact tree."""

# MLflow 3.x leaves some SDK attributes and parameters unannotated.
# pyright: reportUnknownMemberType=false

from pathlib import Path, PurePosixPath
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import cast

from mlflow import MlflowClient

from evolucio.persistence.local import RunManifest
from evolucio.persistence.local.artifacts import artifact_path, sha256_file

from .errors import MlflowTrackingIntegrityError

_TRACKED_FILES = {
    "summary.json": "metadata/summary.json",
    "error.json": "metadata/error.json",
    "artifacts/parquet/datasets.json": "indexes/parquet-datasets.json",
    "artifacts/zarr/index.json": "indexes/zarr-index.json",
    "artifacts/genomes/index.json": "indexes/genomes-index.json",
}


def artifact_sources(run_dir: Path, manifest: RunManifest) -> dict[str, Path]:
    projection = {"metadata/manifest.json": run_dir / "manifest.json"}
    allowed = _TRACKED_FILES | {manifest.config_artifact: "metadata/config.json"}
    for record in manifest.artifacts:
        if record.relative_path in allowed:
            path = artifact_path(run_dir, record.relative_path)
            if not path.is_file() or sha256_file(path) != record.sha256:
                raise MlflowTrackingIntegrityError("canonical metadata artifact digest differs")
            projection[allowed[record.relative_path]] = path
    if "indexes/genomes-index.json" in projection:
        from evolucio.persistence.genomes import read_genome_index

        for genome in read_genome_index(run_dir).genomes:
            path = artifact_path(run_dir, genome.relative_path)
            if not path.is_file() or sha256_file(path) != genome.artifact_sha256:
                raise MlflowTrackingIntegrityError("highlighted genome file digest differs")
            projection[f"genomes/{path.name}"] = path
    return projection


def existing_artifacts(client: MlflowClient, run_id: str, targets: set[str]) -> set[str]:
    directories = {str(PurePosixPath(target).parent) for target in targets}
    return {
        cast(str, item.path)
        for directory in directories
        for item in client.list_artifacts(run_id, directory)
        if not item.is_dir and item.path in targets
    }


def upload_artifact(client: MlflowClient, run_id: str, target: str, source: Path) -> None:
    path = PurePosixPath(target)
    if path.name == source.name:
        client.log_artifact(run_id, str(source), str(path.parent))
    else:
        with TemporaryDirectory() as directory:
            renamed = Path(directory) / path.name
            copyfile(source, renamed)
            client.log_artifact(run_id, str(renamed), str(path.parent))
