"""Independent frozen checkpoint and analytical snapshot contracts."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evolucio.host.schema import HOST_DTO_SCHEMA_DIGEST

CHECKPOINT_SCHEMA_NAME = "simulation_checkpoint_zarr_v1"
CHECKPOINT_SCHEMA_VERSION = 1
SNAPSHOT_SCHEMA_NAME = "analysis_snapshot_zarr_v1"
SNAPSHOT_SCHEMA_VERSION = 1

POPULATION_FIELDS = (
    "alive",
    "agent_id",
    "parent_id",
    "lineage_id",
    "genome_id",
    "generation",
    "position",
    "energy",
    "birth_step",
    "age",
)
SNAPSHOT_POPULATION_FIELDS = ("slot_index", *POPULATION_FIELDS[1:])
WORLD_FIELDS = ("resources", "environment", "occupancy")
ID_FIELDS = ("next_agent_id", "next_genome_id", "next_lineage_id")
GENOME_SHAPES = {
    "layer1_weight": (16, 15),
    "layer1_bias": (16,),
    "layer2_weight": (7, 16),
    "layer2_bias": (7,),
}
ACCUMULATOR_FIELDS = (
    "steps_accumulated",
    "births_total",
    "birth_rejected_capacity_total",
    "deaths_total",
    "deaths_by_cause_total",
    "resource_consumed_total",
    "feeding_energy_gained_total",
    "basal_energy_cost_total",
    "movement_energy_cost_total",
    "feeding_energy_cost_total",
    "reproduction_energy_cost_total",
    "death_energy_removed_total",
    "movement_success_total",
    "feeding_success_total",
    "reproduction_success_total",
    "death_age_sum",
    "alive_count_sum",
    "energy_alive_sum",
    "count_overflow",
)
ACCUMULATOR_FLOATS = frozenset(
    (
        "resource_consumed_total",
        "feeding_energy_gained_total",
        "basal_energy_cost_total",
        "movement_energy_cost_total",
        "feeding_energy_cost_total",
        "reproduction_energy_cost_total",
        "death_energy_removed_total",
        "death_age_sum",
        "energy_alive_sum",
    )
)


@dataclass(frozen=True, slots=True)
class ZarrCapturePolicy:
    checkpoint_interval: int | None = None
    snapshot_interval: int | None = None
    checkpoint_on_completion: bool = False
    snapshot_on_completion: bool = False
    keep_last_checkpoints: int | None = None

    def __post_init__(self) -> None:
        for value in (self.checkpoint_interval, self.snapshot_interval, self.keep_last_checkpoints):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(
                    "capture intervals and retention must be positive integers or None"
                )
        if (
            type(self.checkpoint_on_completion) is not bool
            or type(self.snapshot_on_completion) is not bool
        ):
            raise ValueError("completion capture flags must be bool")

    @property
    def enabled(self) -> bool:
        return bool(
            self.checkpoint_interval
            or self.snapshot_interval
            or self.checkpoint_on_completion
            or self.snapshot_on_completion
        )


DEFAULT_ZARR_POLICY = ZarrCapturePolicy()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CaptureMetadata(StrictModel):
    run_id: str
    state_step: int = Field(ge=0)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compile_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at_utc: str
    package_version: str
    git_commit: str | None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    capacity: int = Field(gt=0)


class CheckpointMetadata(CaptureMetadata):
    schema_name: Literal["simulation_checkpoint_zarr_v1"] = CHECKPOINT_SCHEMA_NAME
    schema_version: Literal[1] = CHECKPOINT_SCHEMA_VERSION
    checkpoint_complete: Literal[True] = True
    rng_implementation: str
    core_schemas: dict[str, str]
    resource_capacity: float = Field(gt=0)


class SnapshotMetadata(CaptureMetadata):
    schema_name: Literal["analysis_snapshot_zarr_v1"] = SNAPSHOT_SCHEMA_NAME
    schema_version: Literal[1] = SNAPSHOT_SCHEMA_VERSION
    snapshot_complete: Literal[True] = True
    host_dto_digest: str = HOST_DTO_SCHEMA_DIGEST
    alive_count: int = Field(ge=0)
    next_agent_id: int = Field(ge=0)
    next_genome_id: int = Field(ge=0)
    next_lineage_id: int = Field(ge=0)
    array_dtypes: dict[str, str]


# Each entry freezes an array path, logical shape and dtype. C/H/W are metadata dimensions.
def checkpoint_array_specs(
    width: int, height: int, capacity: int
) -> dict[str, tuple[tuple[int, ...], str]]:
    specs: dict[str, tuple[tuple[int, ...], str]] = {"step": ((), "int32")}
    specs.update(
        {
            f"world/{name}": ((height, width), "int32" if name == "occupancy" else "float32")
            for name in WORLD_FIELDS
        }
    )
    specs.update(
        {
            f"population/{name}": (
                (capacity, 2) if name == "position" else (capacity,),
                "bool" if name == "alive" else "float32" if name == "energy" else "int32",
            )
            for name in POPULATION_FIELDS
        }
    )
    specs.update(
        {
            f"genomes/{name}": ((capacity, *shape), "float32")
            for name, shape in GENOME_SHAPES.items()
        }
    )
    specs.update({f"ids/{name}": ((), "int32") for name in ID_FIELDS})
    specs["rng/key_data"] = ((2,), "uint32")
    specs.update(
        {
            f"accumulator/{name}": (
                (6,) if name == "deaths_by_cause_total" else (),
                "bool"
                if name == "count_overflow"
                else "float32"
                if name in ACCUMULATOR_FLOATS
                else "int32",
            )
            for name in ACCUMULATOR_FIELDS
        }
    )
    return specs


def validate_capture_header(
    metadata: dict[str, object], *, name: str, version: int, completion_field: str
) -> None:
    """Require an explicit version and completion marker; never infer a legacy schema."""
    if (
        metadata.get("schema_name") != name
        or type(metadata.get("schema_version")) is not int
        or metadata.get("schema_version") != version
        or metadata.get(completion_field) is not True
    ):
        raise ValueError(f"unsupported or incomplete capture schema: {name}")
