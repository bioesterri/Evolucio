"""Classify differences using the current validated configuration schema."""

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from evolucio.config import ExperimentConfig
from evolucio.config.freeze import canonical_json_and_hash


class DifferenceCategory(StrEnum):
    SCIENTIFIC = "SCIENTIFIC"
    OPERATIONAL = "OPERATIONAL"
    SEED = "SEED"
    UNKNOWN = "UNKNOWN"


NEUTRAL_FIELDS = frozenset({"runtime.chunk_size", "persistence.output_dir"})
CAPTURE_FIELDS = frozenset(
    {
        "runtime.record_stride",
        "runtime.snapshot_stride",
        "persistence.level",
        "persistence.destinations",
        "persistence.batch_size",
        "persistence.checkpoint_stride",
    }
)
SCIENTIFIC_FIELDS = frozenset(
    [
        "schema_version",
        "world.width",
        "world.height",
        "world.boundary_mode",
        "world.resource_capacity",
        "world.initial_resource_mean",
        "world.resource_distribution",
        "world.resource_patch_count",
        "world.resource_patch_radius",
        "world.resource_patch_contrast",
        "world.environment_initial_value",
        "world.regeneration_rate",
        "world.environment_schedule",
        "population.initial_agents",
        "population.max_agents",
        "population.max_births_per_step",
        "population.placement",
        "population.allow_multiple_agents_per_cell",
        "policy.action_schema_version",
        "policy.schema_version",
        "policy.input_size",
        "policy.hidden_size",
        "policy.output_size",
        "policy.activation",
        "policy.use_bias",
        "genome.schema_version",
        "genome.initialization",
        "observations.schema_version",
        "observations.perception_radius",
        "energy.initial_energy",
        "energy.max_energy",
        "energy.death_threshold",
        "energy.basal_cost",
        "energy.movement_cost",
        "energy.feeding_cost",
        "energy.feeding_conversion",
        "energy.feeding_max_resource_intake",
        "energy.reproduction_threshold",
        "energy.reproduction_cost",
        "energy.offspring_initial_energy",
        "energy.failed_action_cost",
        "evolution.min_reproduction_age",
        "evolution.max_age",
        "evolution.weight_mutation_rate",
        "evolution.weight_mutation_sigma",
        "evolution.weight_abs_limit",
        "evolution.bias_mutation_rate",
        "evolution.bias_mutation_sigma",
        "evolution.bias_abs_limit",
        "runtime.steps",
        "runtime.backend",
    ]
)


@dataclass(frozen=True, slots=True)
class ConfigDiff:
    parameter_path: str
    baseline_value: object
    variant_value: object
    difference_category: DifferenceCategory


def _flatten(config: ExperimentConfig) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in config.model_dump(mode="json").items():
        if isinstance(value, dict):
            result.update(
                {f"{key}.{name}": item for name, item in cast(dict[str, object], value).items()}
            )
        else:
            result[key] = value
    return result


def _category(path: str) -> DifferenceCategory:
    if path == "seed":
        return DifferenceCategory.SEED
    if path in SCIENTIFIC_FIELDS:
        return DifferenceCategory.SCIENTIFIC
    if path in NEUTRAL_FIELDS | CAPTURE_FIELDS:
        return DifferenceCategory.OPERATIONAL
    return DifferenceCategory.UNKNOWN


def compare_configs(
    baseline: ExperimentConfig, variant: ExperimentConfig
) -> tuple[ConfigDiff, ...]:
    left, right = _flatten(baseline), _flatten(variant)
    return tuple(
        ConfigDiff(path, left.get(path), right.get(path), _category(path))
        for path in sorted(left.keys() | right.keys())
        if path not in left or path not in right or left[path] != right[path]
    )


def configuration_group_hash(config: ExperimentConfig) -> str:
    """Exclude the master seed and the explicitly neutral fields only."""
    return canonical_json_and_hash(
        {
            path: value
            for path, value in _flatten(config).items()
            if path != "seed" and path not in NEUTRAL_FIELDS
        }
    )[1]
