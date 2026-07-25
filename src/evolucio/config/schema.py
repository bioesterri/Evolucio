"""Generated JSON Schema API."""

from .models import ExperimentConfig


def experiment_config_json_schema() -> dict[str, object]:
    """Return JSON Schema derived from the authoritative Pydantic model."""
    return ExperimentConfig.model_json_schema()
