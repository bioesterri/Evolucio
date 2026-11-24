"""Scientifically relevant scalar parameters from the frozen host configuration."""

from evolucio.config import ExperimentConfig
from evolucio.persistence.local import RunManifest


def build_mlflow_params(config: ExperimentConfig, run_metadata: RunManifest) -> dict[str, str]:
    params = {
        "seed": str(config.seed),
        "total_steps": str(run_metadata.total_steps),
        "chunk_steps": str(run_metadata.chunk_steps),
        "initial_agents": str(config.population.initial_agents),
        "max_agents": str(config.population.max_agents),
    }
    for name, section in (
        ("world", config.world),
        ("population", config.population),
        ("energy", config.energy),
        ("evolution", config.evolution),
        ("policy", config.policy),
        ("observations", config.observations),
    ):
        for key, value in section.model_dump().items():
            if type(value) in (bool, int, float, str):
                params[f"{name}.{key}"] = str(value)
    return params
