"""Stable experiment lookup, recovery, and PostgreSQL binding creation."""

# MLflow 3.x leaves some SDK attributes and parameters unannotated.
# pyright: reportUnknownMemberType=false

import re
from datetime import UTC, datetime
from uuid import UUID

from mlflow import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException

from evolucio.persistence.postgres.models import MlflowExperimentBindingModel
from evolucio.persistence.postgres.repository import PostgresRegistry

from .errors import MlflowExperimentBindingError, MlflowTrackingIntegrityError

EXPERIMENT_TAG = "evolucio.experiment_id"


def get_or_create_mlflow_experiment(
    registry: PostgresRegistry, client: MlflowClient, experiment_id: UUID
) -> MlflowExperimentBindingModel:
    try:
        with registry.lock_mlflow_experiment(experiment_id) as (session, experiment):
            binding = session.get(MlflowExperimentBindingModel, experiment_id)
            if binding is not None:
                remote = client.get_experiment(binding.mlflow_experiment_id)
                if remote.lifecycle_stage != "active" or remote.tags.get(EXPERIMENT_TAG) != str(
                    experiment_id
                ):
                    raise MlflowTrackingIntegrityError("bound MLflow experiment identity differs")
                return binding

            tags = {EXPERIMENT_TAG: str(experiment_id), "evolucio.project": "Evolucio"}
            matches = client.search_experiments(
                view_type=ViewType.ALL,
                filter_string=f"tags.`{EXPERIMENT_TAG}` = '{experiment_id}'",
                max_results=2,
            )
            if len(matches) > 1 or matches.token:
                raise MlflowTrackingIntegrityError("multiple MLflow experiments match the UUID")
            if matches:
                remote = matches[0]
            else:
                slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", experiment.name).strip("-")[:60]
                name = f"Evolucio/{slug or 'experiment'}-{experiment_id}"
                existing = client.get_experiment_by_name(name)
                if existing is None:
                    try:
                        remote_id = client.create_experiment(name, tags=tags)
                    except MlflowException as exc:
                        existing = client.get_experiment_by_name(name)
                        if existing is None:
                            raise MlflowExperimentBindingError(
                                "could not create the MLflow experiment"
                            ) from exc
                        remote = existing
                    else:
                        remote = client.get_experiment(remote_id)
                else:
                    remote = existing
            if remote.lifecycle_stage != "active" or remote.tags.get(EXPERIMENT_TAG) != str(
                experiment_id
            ):
                raise MlflowTrackingIntegrityError("MLflow experiment name or identity conflict")
            binding = MlflowExperimentBindingModel(
                experiment_id=experiment_id,
                mlflow_experiment_id=remote.experiment_id,
                mlflow_experiment_name=remote.name,
                created_at=datetime.now(UTC),
            )
            session.add(binding)
            return binding
    except (MlflowExperimentBindingError, MlflowTrackingIntegrityError):
        raise
    except Exception as exc:
        raise MlflowExperimentBindingError("could not bind the MLflow experiment") from exc
