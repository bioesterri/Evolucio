"""Strict, immutable host configuration models."""

from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
Fraction = Annotated[float, Field(ge=0, le=1)]
Rate = Annotated[float, Field(ge=0, lt=1)]
_STEP_MAX = 2**31 - 1


class _ConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class EnvironmentPhaseConfig(_ConfigModel):
    """A deterministic environmental interval, with an exclusive end."""

    start_step: Annotated[int, Field(ge=0, le=_STEP_MAX)]
    end_step: Annotated[int, Field(gt=0, le=_STEP_MAX)]
    regeneration_multiplier: Fraction
    stress_level: Fraction

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end_step <= self.start_step:
            raise ValueError("end_step must be greater than start_step")
        return self


class WorldConfig(_ConfigModel):
    """Declarative two-dimensional world parameters."""

    width: PositiveInt
    height: PositiveInt
    boundary_mode: Literal["closed"]
    resource_capacity: NonNegativeFloat
    initial_resource_mean: NonNegativeFloat
    resource_distribution: Literal["uniform", "patches"]
    resource_patch_count: PositiveInt
    resource_patch_radius: PositiveFloat
    resource_patch_contrast: Fraction
    environment_initial_value: Fraction
    regeneration_rate: Rate
    environment_schedule: tuple[EnvironmentPhaseConfig, ...] = ()

    @model_validator(mode="after")
    def validate_initial_resource(self) -> Self:
        if self.initial_resource_mean > self.resource_capacity:
            raise ValueError("initial_resource_mean must not exceed resource_capacity")
        if self.width * self.height > _STEP_MAX:
            raise ValueError("world area must be representable as int32")
        return self

    @field_validator("environment_schedule", mode="before")
    @classmethod
    def freeze_schedule(cls, value: object) -> object:
        return tuple(cast(list[object], value)) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        previous: EnvironmentPhaseConfig | None = None
        for index, phase in enumerate(self.environment_schedule):
            if previous is not None and phase.start_step < previous.start_step:
                raise ValueError(
                    f"environment_schedule[{index}].start_step must be strictly ordered"
                )
            if previous is not None and phase.start_step < previous.end_step:
                raise ValueError(f"environment_schedule[{index}].start_step must not overlap")
            previous = phase
        return self


class PopulationConfig(_ConfigModel):
    """Fixed-capacity population parameters."""

    initial_agents: NonNegativeInt
    max_agents: PositiveInt
    max_births_per_step: PositiveInt
    placement: Literal["random"]
    allow_multiple_agents_per_cell: bool

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.initial_agents > self.max_agents:
            raise ValueError("initial_agents must not exceed max_agents")
        if self.max_births_per_step > self.max_agents:
            raise ValueError("max_births_per_step must not exceed max_agents")
        return self


class PolicyConfig(_ConfigModel):
    """Versioned fixed-topology policy parameters."""

    action_schema_version: Literal["1.0"]
    schema_version: Literal[1] = 1
    input_size: Literal[15] = 15
    hidden_size: Literal[16] = 16
    output_size: Literal[7] = 7
    activation: Literal["tanh"] = "tanh"
    use_bias: Literal[True] = True


class GenomeConfig(_ConfigModel):
    """Versioned founder-genome initialization selector."""

    schema_version: Literal[1] = 1
    initialization: Literal["glorot_uniform_zero_bias_v1"] = "glorot_uniform_zero_bias_v1"


class ObservationsConfig(_ConfigModel):
    """Fixed local observation schema and configurable static radius."""

    schema_version: Literal[1] = 1
    perception_radius: Annotated[int, Field(ge=1, le=3)] = 1


class EnergyConfig(_ConfigModel):
    """Energy economy parameters; no runtime balance is implemented here."""

    initial_energy: float
    max_energy: float
    death_threshold: NonNegativeFloat
    basal_cost: NonNegativeFloat
    movement_cost: NonNegativeFloat
    feeding_cost: NonNegativeFloat
    feeding_conversion: PositiveFloat
    feeding_max_resource_intake: PositiveFloat
    reproduction_threshold: float
    reproduction_cost: NonNegativeFloat
    offspring_initial_energy: float
    failed_action_cost: NonNegativeFloat

    @model_validator(mode="after")
    def validate_energy(self) -> Self:
        if not self.death_threshold < self.initial_energy <= self.max_energy:
            raise ValueError("initial_energy must be above death_threshold and at most max_energy")
        if self.max_energy <= self.death_threshold:
            raise ValueError("max_energy must be above death_threshold")
        if self.reproduction_threshold > self.max_energy:
            raise ValueError("reproduction_threshold must not exceed max_energy")
        if self.offspring_initial_energy <= self.death_threshold:
            raise ValueError("offspring_initial_energy must be above death_threshold")
        minimum = self.death_threshold + self.reproduction_cost + self.offspring_initial_energy
        if self.reproduction_threshold <= minimum:
            raise ValueError("reproduction_threshold does not leave the parent viable")
        return self


class EvolutionConfig(_ConfigModel):
    """Age and heritable mutation parameters."""

    min_reproduction_age: Annotated[int, Field(ge=1)]
    max_age: PositiveInt
    mutation_rate: Fraction
    mutation_sigma: NonNegativeFloat
    mutation_clip_abs: PositiveFloat

    @model_validator(mode="after")
    def validate_ages(self) -> Self:
        if self.max_age <= self.min_reproduction_age:
            raise ValueError("max_age must be greater than min_reproduction_age")
        return self


class RuntimeConfig(_ConfigModel):
    """Host execution parameters."""

    steps: PositiveInt
    chunk_size: PositiveInt
    record_stride: PositiveInt
    snapshot_stride: PositiveInt | None
    backend: Literal["cpu", "auto"]


class PersistenceConfig(_ConfigModel):
    """Host-only persistence intent."""

    level: Literal["none", "minimal", "standard", "massive"] = "none"
    destinations: tuple[Literal["local", "postgresql", "mlflow"], ...] = ()
    output_dir: Annotated[str, Field(min_length=1)] = "runs"
    batch_size: PositiveInt = 1024
    checkpoint_stride: PositiveInt | None = None

    @field_validator("destinations", mode="before")
    @classmethod
    def freeze_destinations(cls, value: object) -> object:
        return tuple(cast(list[object], value)) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_persistence(self) -> Self:
        if len(set(self.destinations)) != len(self.destinations):
            raise ValueError("destinations must not contain duplicates")
        if self.level == "none" and (self.destinations or self.checkpoint_stride is not None):
            raise ValueError("level none requires no destinations and no checkpoint stride")
        if self.level != "none" and not self.destinations:
            raise ValueError("enabled persistence requires at least one destination")
        return self


class ExperimentConfig(_ConfigModel):
    """Complete validated scientific configuration for schema 1.3."""

    schema_version: Literal["1.3"]
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)]
    world: WorldConfig
    population: PopulationConfig
    policy: PolicyConfig
    genome: GenomeConfig = Field(default_factory=GenomeConfig)
    observations: ObservationsConfig = Field(default_factory=ObservationsConfig)
    energy: EnergyConfig
    evolution: EvolutionConfig
    runtime: RuntimeConfig
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)

    @model_validator(mode="after")
    def validate_blocks(self) -> Self:
        if not self.population.allow_multiple_agents_per_cell:
            area = self.world.width * self.world.height
            if self.population.max_agents > area or self.population.initial_agents > area:
                raise ValueError("population capacity exceeds world area without cell sharing")
        return self
