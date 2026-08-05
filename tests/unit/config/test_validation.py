import pytest
from pydantic import ValidationError

from evolucio.config import ExperimentConfig


def raw(config: ExperimentConfig) -> dict[str, object]:
    return config.model_dump(mode="python")


def test_environment_schedule(config: ExperimentConfig) -> None:
    value = raw(config)
    value["world"]["environment_schedule"] = [  # type: ignore[index]
        {"start_step": 0, "end_step": 10, "regeneration_multiplier": 1.0, "stress_level": 0.0},
        {"start_step": 10, "end_step": 20, "regeneration_multiplier": 0.5, "stress_level": 0.2},
    ]
    assert len(ExperimentConfig.model_validate(value).world.environment_schedule) == 2
    value["world"]["environment_schedule"].reverse()  # type: ignore[index,union-attr]
    with pytest.raises(ValidationError, match="ordered"):
        ExperimentConfig.model_validate(value)


@pytest.mark.parametrize(
    "phases,message",
    [
        (
            [
                {
                    "start_step": 0,
                    "end_step": 10,
                    "regeneration_multiplier": 1.0,
                    "stress_level": 0.0,
                },
                {
                    "start_step": 9,
                    "end_step": 20,
                    "regeneration_multiplier": 1.0,
                    "stress_level": 0.0,
                },
            ],
            "overlap",
        ),
        (
            [{"start_step": 2, "end_step": 2, "regeneration_multiplier": 1.0, "stress_level": 0.0}],
            "greater",
        ),
        (
            [
                {
                    "start_step": 0,
                    "end_step": 2,
                    "regeneration_multiplier": -1.0,
                    "stress_level": 0.0,
                }
            ],
            "regeneration_multiplier",
        ),
        (
            [{"start_step": 0, "end_step": 2, "regeneration_multiplier": 1.0, "stress_level": 2.0}],
            "stress_level",
        ),
    ],
)
def test_bad_schedule(
    config: ExperimentConfig, phases: list[dict[str, object]], message: str
) -> None:
    value = raw(config)
    value["world"]["environment_schedule"] = phases  # type: ignore[index]
    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(value)


def test_environment_schedule_must_fit_runtime(config: ExperimentConfig) -> None:
    value = raw(config)
    runtime = value["runtime"]  # type: ignore[index]
    runtime["steps"] = 4  # type: ignore[index]
    value["world"]["environment_schedule"] = [  # type: ignore[index]
        {"start_step": 3, "end_step": 5, "regeneration_multiplier": 1.0, "stress_level": 0.0}
    ]
    with pytest.raises(ValidationError, match=r"environment_schedule\[0\]\.end_step"):
        ExperimentConfig.model_validate(value)

    value["world"]["environment_schedule"] = [  # type: ignore[index]
        {"start_step": 4, "end_step": 5, "regeneration_multiplier": 1.0, "stress_level": 0.0}
    ]
    with pytest.raises(ValidationError, match=r"environment_schedule\[0\]\.end_step"):
        ExperimentConfig.model_validate(value)


def test_persistence_rules(config: ExperimentConfig) -> None:
    value = raw(config)
    value["persistence"]["destinations"] = ("local",)  # type: ignore[index]
    with pytest.raises(ValidationError, match="level none"):
        ExperimentConfig.model_validate(value)
    value["persistence"] = {
        "level": "standard",
        "destinations": (),
        "output_dir": "runs",
        "batch_size": 1,
        "checkpoint_stride": None,
    }
    with pytest.raises(ValidationError, match="destination"):
        ExperimentConfig.model_validate(value)
    value["persistence"]["destinations"] = ("local", "local")  # type: ignore[index]
    with pytest.raises(ValidationError, match="duplicates"):
        ExperimentConfig.model_validate(value)


@pytest.mark.parametrize(
    "genome",
    [
        {"schema_version": 2},
        {"initialization": "unknown"},
        {"initialization": ""},
    ],
)
def test_invalid_genome_contract_is_rejected(
    config: ExperimentConfig, genome: dict[str, object]
) -> None:
    candidate = config.model_dump(mode="python")
    candidate["genome"] = genome
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(candidate)
