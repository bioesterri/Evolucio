import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from evolucio.config import ExperimentConfig


def data(config: ExperimentConfig) -> dict[str, object]:
    return config.model_dump(mode="python")


def error(config: ExperimentConfig, path: tuple[str, ...], value: object) -> ValidationError:
    raw = data(config)
    current = raw
    for key in path[:-1]:
        current = current[key]  # type: ignore[index,assignment]
    current[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValidationError) as caught:
        ExperimentConfig.model_validate(raw)
    return caught.value


def test_valid_and_frozen(config: ExperimentConfig) -> None:
    assert config.seed == 42 and isinstance(config.world.environment_schedule, tuple)
    with pytest.raises(ValidationError, match="frozen"):
        config.seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        (("seed",), -1, "seed"),
        (("seed",), 2**32, "seed"),
        (("seed",), True, "seed"),
        (("world", "width"), 0, "width"),
        (("world", "resource_capacity"), math.inf, "resource_capacity"),
        (("world", "initial_resource_mean"), 10.1, "initial_resource_mean"),
        (("world", "resource_patch_count"), 0, "resource_patch_count"),
        (("world", "resource_patch_count"), True, "resource_patch_count"),
        (("world", "resource_patch_radius"), 0.0, "resource_patch_radius"),
        (("world", "resource_patch_radius"), 0.249, "resource_patch_radius"),
        (("world", "resource_patch_radius"), math.nan, "resource_patch_radius"),
        (("world", "resource_patch_contrast"), -0.1, "resource_patch_contrast"),
        (("world", "resource_patch_contrast"), 1.1, "resource_patch_contrast"),
        (("world", "environment_initial_value"), -0.1, "environment_initial_value"),
        (("world", "environment_initial_value"), math.inf, "environment_initial_value"),
        (("world", "boundary_mode"), "toroidal", "boundary_mode"),
        (("world", "resource_distribution"), "random", "resource_distribution"),
        (("population", "initial_agents"), 2000, "population"),
        (("policy", "hidden_size"), 8, "hidden_size"),
        (("observations", "schema_version"), 2, "schema_version"),
        (("observations", "perception_radius"), True, "perception_radius"),
        (("observations", "perception_radius"), 0, "perception_radius"),
        (("observations", "perception_radius"), 4, "perception_radius"),
        (("energy", "initial_energy"), 101.0, "energy"),
        (("energy", "feeding_max_resource_intake"), 0.0, "feeding_max_resource_intake"),
        (("energy", "feeding_max_resource_intake"), True, "feeding_max_resource_intake"),
        (("energy", "reproduction_threshold"), 15.0, "energy"),
        (("evolution", "max_age"), 5, "evolution"),
        (("evolution", "mutation_rate"), 1.2, "mutation_rate"),
        (("runtime", "record_stride"), 0, "record_stride"),
    ],
)
def test_invalid_values(
    config: ExperimentConfig, path: tuple[str, ...], value: object, field: str
) -> None:
    assert field in str(error(config, path, value))


def test_resource_patch_radius_accepts_declared_minimum(config: ExperimentConfig) -> None:
    raw = data(config)
    raw["world"]["resource_patch_radius"] = 0.25  # type: ignore[index]
    parsed = ExperimentConfig.model_validate(raw)
    assert parsed.world.resource_patch_radius == 0.25


def test_extra_and_string_coercion(config: ExperimentConfig) -> None:
    raw = data(config)
    raw["unknown"] = 1
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExperimentConfig.model_validate(raw)
    assert "int_type" in str(error(config, ("seed",), "10"))


def test_cross_block_capacity(config: ExperimentConfig) -> None:
    raw = data(config)
    raw["world"]["width"] = 2
    raw["world"]["height"] = 2  # type: ignore[index]
    raw["population"]["allow_multiple_agents_per_cell"] = False  # type: ignore[index]
    with pytest.raises(ValidationError, match="world area"):
        ExperimentConfig.model_validate(raw)


def test_zero_founders_and_overlapping_population_are_valid(config: ExperimentConfig) -> None:
    raw = data(config)
    raw["world"]["width"] = 1  # type: ignore[index]
    raw["world"]["height"] = 1  # type: ignore[index]
    raw["population"]["initial_agents"] = 0  # type: ignore[index]
    raw["population"]["max_agents"] = 4  # type: ignore[index]
    raw["population"]["max_births_per_step"] = 1  # type: ignore[index]
    assert ExperimentConfig.model_validate(raw).population.initial_agents == 0


def test_world_area_must_fit_linear_int32_index(config: ExperimentConfig) -> None:
    raw = data(config)
    raw["world"]["width"] = 46341  # type: ignore[index]
    raw["world"]["height"] = 46341  # type: ignore[index]
    with pytest.raises(ValidationError, match="representable as int32"):
        ExperimentConfig.model_validate(raw)


@given(st.integers(min_value=0, max_value=2**32 - 1))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_valid_seeds(config: ExperimentConfig, seed: int) -> None:
    raw = data(config)
    raw["seed"] = seed
    assert ExperimentConfig.model_validate(raw).seed == seed


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("input_size", 14),
        ("input_size", True),
        ("hidden_size", 15),
        ("output_size", 8),
        ("activation", "relu"),
        ("use_bias", False),
    ],
)
def test_policy_contract_rejects_alternatives(
    config: ExperimentConfig, field: str, value: object
) -> None:
    assert field in str(error(config, ("policy", field), value))
