"""Explicit tracking client construction and read-only health checks."""

# MLflow 3.x leaves some SDK attributes and parameters unannotated.
# pyright: reportUnknownMemberType=false

import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from mlflow import MlflowClient

from .errors import MlflowUnavailableError


def create_tracking_client(tracking_uri: str | None = None) -> MlflowClient:
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise MlflowUnavailableError("MLFLOW_TRACKING_URI is not configured")
    try:
        return MlflowClient(tracking_uri=uri)
    except Exception as exc:
        raise MlflowUnavailableError("could not initialize the MLflow tracking client") from exc


@dataclass(frozen=True, slots=True)
class MlflowConnectionStatus:
    configured: bool
    connected: bool
    version: str | None


def check_mlflow_connection(tracking_uri: str | None = None) -> MlflowConnectionStatus:
    configured = bool(tracking_uri or os.getenv("MLFLOW_TRACKING_URI"))
    try:
        package_version = version("mlflow")
    except PackageNotFoundError:
        package_version = None
    try:
        create_tracking_client(tracking_uri).search_experiments(max_results=1)
    except Exception:
        return MlflowConnectionStatus(configured, False, package_version)
    return MlflowConnectionStatus(configured, True, package_version)
