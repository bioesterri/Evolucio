"""Compile validated host configuration into the JAX-facing contract."""

import dataclasses
import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.actions import (
    ACTION_CONTRACT_SCHEMA_DIGEST,
    ACTION_CONTRACT_SCHEMA_VERSION,
    FEEDING_RESOLUTION_SCHEMA_DIGEST,
    FEEDING_RESOLUTION_SCHEMA_VERSION,
    MOVEMENT_RESOLUTION_SCHEMA_DIGEST,
    MOVEMENT_RESOLUTION_SCHEMA_VERSION,
)
from evolucio.core.dtypes import REAL_DTYPE, STEP_DTYPE
from evolucio.core.observations.schema import (
    OBSERVATION_SCHEMA_DIGEST,
    OBSERVATION_SIZE,
)
from evolucio.core.policy import (
    ACTION_SELECTION_SCHEMA_DIGEST,
    ACTION_SELECTION_SCHEMA_VERSION,
    GENOME_INITIALIZATION_NAME,
    GENOME_INITIALIZATION_VERSION,
    GENOME_PARAMETER_COUNT,
    GENOME_SCHEMA_DIGEST,
    POLICY_SCHEMA_DIGEST,
)
from evolucio.core.rng import PRNG_IMPLEMENTATION

from .freeze import canonical_json_and_hash, freeze_config
from .models import ExperimentConfig

COMPILE_SIGNATURE_SCHEMA_VERSION = 10
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_FLOAT32_MAX = 3.4028235e38


class ConfigCompilationError(ValueError):
    """A validated host value cannot be represented by the core contract."""


@dataclass(frozen=True, slots=True)
class CompileSignature:
    """Immutable allowlist of values that determine a compiled executable."""

    signature_schema_version: int
    config_schema_version: str
    world_width: int
    world_height: int
    boundary_mode: str
    resource_distribution: str
    resource_patch_count: int
    environment_schedule_length: int
    max_agents: int
    max_births_per_step: int
    placement: str
    allow_multiple_agents_per_cell: bool
    observation_schema_version: int
    observation_schema_size: int
    observation_schema_digest: str
    action_schema_version: str
    policy_schema_version: int
    policy_schema_digest: str
    policy_input_size: int
    policy_hidden_size: int
    policy_output_size: int
    policy_activation: str
    policy_use_bias: bool
    action_selection_schema_version: int
    action_selection_schema_digest: str
    action_count: int
    action_contract_schema_version: int
    action_contract_schema_digest: str
    movement_resolution_schema_version: int
    movement_resolution_schema_digest: str
    feeding_resolution_schema_version: int
    feeding_resolution_schema_digest: str
    genome_schema_version: int
    genome_schema_digest: str
    genome_initialization_name: str
    genome_initialization_version: int
    genome_parameter_count: int
    perception_radius: int
    chunk_size: int
    record_stride: int
    snapshot_stride: int | None
    backend: str
    rng_implementation: str


class WorldCoreConfig(eqx.Module):
    """World parameters used by compiled core functions."""

    width: int
    height: int
    boundary_mode: str
    resource_distribution: str
    resource_capacity: jax.Array
    initial_resource_mean: jax.Array
    resource_patch_count: int
    resource_patch_radius: jax.Array
    resource_patch_contrast: jax.Array
    environment_initial_value: jax.Array
    regeneration_rate: jax.Array
    environment_calendar: "EnvironmentCalendarCoreConfig"


class EnvironmentCalendarCoreConfig(eqx.Module):
    """Fixed-length environmental calendar consumed by compiled world updates."""

    phase_count: int = eqx.field(static=True)
    start_steps: jax.Array
    end_steps: jax.Array
    regeneration_multipliers: jax.Array
    environment_values: jax.Array


class PopulationCoreConfig(eqx.Module):
    """Population parameters used by compiled core functions."""

    initial_agents: jax.Array
    max_agents: int
    max_births_per_step: int
    placement: str
    allow_multiple_agents_per_cell: bool


class PolicyCoreConfig(eqx.Module):
    """Fixed policy topology and schema selectors."""

    action_schema_version: str = eqx.field(static=True)
    schema_version: int = eqx.field(static=True)
    schema_digest: str = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    activation: str = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    action_selection_schema_version: int = eqx.field(static=True)
    action_selection_schema_digest: str = eqx.field(static=True)


class ObservationsCoreConfig(eqx.Module):
    """Static, hashable local observation contract."""

    schema_version: int = eqx.field(static=True)
    schema_size: int = eqx.field(static=True)
    schema_digest: str = eqx.field(static=True)
    perception_radius: int = eqx.field(static=True)


class GenomeCoreConfig(eqx.Module):
    """Static, hashable neural-genome contract."""

    schema_version: int = eqx.field(static=True)
    schema_digest: str = eqx.field(static=True)
    initialization_name: str = eqx.field(static=True)
    initialization_version: int = eqx.field(static=True)
    parameter_count: int = eqx.field(static=True)


class EnergyCoreConfig(eqx.Module):
    """Dynamic energy economy parameters."""

    initial_energy: jax.Array
    max_energy: jax.Array
    death_threshold: jax.Array
    basal_cost: jax.Array
    movement_cost: jax.Array
    feeding_cost: jax.Array
    feeding_conversion: jax.Array
    feeding_max_resource_intake: jax.Array
    reproduction_threshold: jax.Array
    reproduction_cost: jax.Array
    offspring_initial_energy: jax.Array
    failed_action_cost: jax.Array


class EvolutionCoreConfig(eqx.Module):
    """Dynamic age and mutation parameters."""

    min_reproduction_age: jax.Array
    max_age: jax.Array
    mutation_rate: jax.Array
    mutation_sigma: jax.Array
    mutation_clip_abs: jax.Array


class RuntimeCoreConfig(eqx.Module):
    """Static output and chunk controls read at the compiled boundary."""

    chunk_size: int
    record_stride: int
    snapshot_stride: int | None


class CoreConfig(eqx.Module):
    """JAX-compatible projection of configuration needed by the core."""

    world: WorldCoreConfig
    population: PopulationCoreConfig
    policy: PolicyCoreConfig
    observations: ObservationsCoreConfig
    genome: GenomeCoreConfig
    energy: EnergyCoreConfig
    evolution: EvolutionCoreConfig
    runtime: RuntimeCoreConfig


@dataclass(frozen=True, slots=True)
class CompiledConfig:
    """Core projection together with run and executable identities."""

    core: CoreConfig
    compile_signature: CompileSignature
    config_hash: str


def _static_int(value: int, field: str) -> int:
    if not _INT32_MIN <= value <= _INT32_MAX:
        raise ConfigCompilationError(f"{field} is outside the supported int32 range")
    return value


def _int_scalar(value: int, field: str) -> jax.Array:
    _static_int(value, field)
    result = jnp.asarray(value, dtype=STEP_DTYPE)  # pyright: ignore[reportUnknownMemberType]
    if result.shape != ():
        raise ConfigCompilationError(f"{field} must compile to a scalar")
    return result


def _float_scalar(value: float, field: str) -> jax.Array:
    if not math.isfinite(value):
        raise ConfigCompilationError(f"{field} must be finite")
    if abs(value) > _FLOAT32_MAX:
        raise ConfigCompilationError(f"{field} is outside the supported float32 range")
    result = jnp.asarray(value, dtype=REAL_DTYPE)  # pyright: ignore[reportUnknownMemberType]
    if result.shape != () or not math.isfinite(float(result)):
        raise ConfigCompilationError(f"{field} must compile to a finite scalar")
    return result


def _int_vector(values: tuple[int, ...], field: str) -> jax.Array:
    for value in values:
        _static_int(value, field)
    return jnp.asarray(values, dtype=STEP_DTYPE)  # pyright: ignore[reportUnknownMemberType]


def _float_vector(values: tuple[float, ...], field: str) -> jax.Array:
    for value in values:
        if not math.isfinite(value) or abs(value) > _FLOAT32_MAX:
            raise ConfigCompilationError(f"{field} contains a value outside the float32 range")
    result = jnp.asarray(values, dtype=REAL_DTYPE)  # pyright: ignore[reportUnknownMemberType]
    if not bool(jnp.all(jnp.isfinite(result))):
        raise ConfigCompilationError(f"{field} must contain only finite values")
    return result


def build_compile_signature(config: ExperimentConfig) -> CompileSignature:
    """Project explicitly allowlisted static host values into a signature."""
    world = config.world
    population = config.population
    policy = config.policy
    runtime = config.runtime
    return CompileSignature(
        signature_schema_version=COMPILE_SIGNATURE_SCHEMA_VERSION,
        config_schema_version=config.schema_version,
        world_width=_static_int(world.width, "world.width"),
        world_height=_static_int(world.height, "world.height"),
        boundary_mode=world.boundary_mode,
        resource_distribution=world.resource_distribution,
        resource_patch_count=_static_int(world.resource_patch_count, "world.resource_patch_count"),
        environment_schedule_length=_static_int(
            len(world.environment_schedule), "world.environment_schedule"
        ),
        max_agents=_static_int(population.max_agents, "population.max_agents"),
        max_births_per_step=_static_int(
            population.max_births_per_step, "population.max_births_per_step"
        ),
        placement=population.placement,
        allow_multiple_agents_per_cell=population.allow_multiple_agents_per_cell,
        observation_schema_version=config.observations.schema_version,
        observation_schema_size=OBSERVATION_SIZE,
        observation_schema_digest=OBSERVATION_SCHEMA_DIGEST,
        action_schema_version=policy.action_schema_version,
        policy_schema_version=policy.schema_version,
        policy_schema_digest=POLICY_SCHEMA_DIGEST,
        policy_input_size=_static_int(policy.input_size, "policy.input_size"),
        policy_hidden_size=_static_int(policy.hidden_size, "policy.hidden_size"),
        policy_output_size=_static_int(policy.output_size, "policy.output_size"),
        policy_activation=policy.activation,
        policy_use_bias=policy.use_bias,
        action_selection_schema_version=ACTION_SELECTION_SCHEMA_VERSION,
        action_selection_schema_digest=ACTION_SELECTION_SCHEMA_DIGEST,
        action_count=_static_int(policy.output_size, "policy.output_size"),
        action_contract_schema_version=ACTION_CONTRACT_SCHEMA_VERSION,
        action_contract_schema_digest=ACTION_CONTRACT_SCHEMA_DIGEST,
        movement_resolution_schema_version=MOVEMENT_RESOLUTION_SCHEMA_VERSION,
        movement_resolution_schema_digest=MOVEMENT_RESOLUTION_SCHEMA_DIGEST,
        feeding_resolution_schema_version=FEEDING_RESOLUTION_SCHEMA_VERSION,
        feeding_resolution_schema_digest=FEEDING_RESOLUTION_SCHEMA_DIGEST,
        genome_schema_version=config.genome.schema_version,
        genome_schema_digest=GENOME_SCHEMA_DIGEST,
        genome_initialization_name=config.genome.initialization,
        genome_initialization_version=GENOME_INITIALIZATION_VERSION,
        genome_parameter_count=GENOME_PARAMETER_COUNT,
        perception_radius=_static_int(
            config.observations.perception_radius, "observations.perception_radius"
        ),
        chunk_size=_static_int(runtime.chunk_size, "runtime.chunk_size"),
        record_stride=_static_int(runtime.record_stride, "runtime.record_stride"),
        snapshot_stride=(
            None
            if runtime.snapshot_stride is None
            else _static_int(runtime.snapshot_stride, "runtime.snapshot_stride")
        ),
        backend=runtime.backend,
        rng_implementation=PRNG_IMPLEMENTATION,
    )


def compile_signature_digest(signature: CompileSignature) -> str:
    """Return a process-independent SHA-256 identity for a compile signature."""
    _, digest = canonical_json_and_hash(dataclasses.asdict(signature))
    return digest


def compile_config(config: ExperimentConfig) -> CompiledConfig:
    """Compile an already validated host model without I/O or side effects."""
    world = config.world
    phases = world.environment_schedule
    core = CoreConfig(
        world=WorldCoreConfig(
            width=_static_int(world.width, "world.width"),
            height=_static_int(world.height, "world.height"),
            boundary_mode=world.boundary_mode,
            resource_distribution=world.resource_distribution,
            resource_capacity=_float_scalar(world.resource_capacity, "world.resource_capacity"),
            initial_resource_mean=_float_scalar(
                world.initial_resource_mean, "world.initial_resource_mean"
            ),
            resource_patch_count=_static_int(
                world.resource_patch_count, "world.resource_patch_count"
            ),
            resource_patch_radius=_float_scalar(
                world.resource_patch_radius, "world.resource_patch_radius"
            ),
            resource_patch_contrast=_float_scalar(
                world.resource_patch_contrast, "world.resource_patch_contrast"
            ),
            environment_initial_value=_float_scalar(
                world.environment_initial_value, "world.environment_initial_value"
            ),
            regeneration_rate=_float_scalar(world.regeneration_rate, "world.regeneration_rate"),
            environment_calendar=EnvironmentCalendarCoreConfig(
                phase_count=len(phases),
                start_steps=_int_vector(
                    tuple(phase.start_step for phase in phases),
                    "world.environment_schedule.start_step",
                ),
                end_steps=_int_vector(
                    tuple(phase.end_step for phase in phases),
                    "world.environment_schedule.end_step",
                ),
                regeneration_multipliers=_float_vector(
                    tuple(phase.regeneration_multiplier for phase in phases),
                    "world.environment_schedule.regeneration_multiplier",
                ),
                environment_values=_float_vector(
                    tuple(phase.stress_level for phase in phases),
                    "world.environment_schedule.stress_level",
                ),
            ),
        ),
        population=PopulationCoreConfig(
            initial_agents=_int_scalar(
                config.population.initial_agents, "population.initial_agents"
            ),
            max_agents=_static_int(config.population.max_agents, "population.max_agents"),
            max_births_per_step=_static_int(
                config.population.max_births_per_step, "population.max_births_per_step"
            ),
            placement=config.population.placement,
            allow_multiple_agents_per_cell=config.population.allow_multiple_agents_per_cell,
        ),
        policy=PolicyCoreConfig(
            **config.policy.model_dump(),
            schema_digest=POLICY_SCHEMA_DIGEST,
            action_selection_schema_version=ACTION_SELECTION_SCHEMA_VERSION,
            action_selection_schema_digest=ACTION_SELECTION_SCHEMA_DIGEST,
        ),
        observations=ObservationsCoreConfig(
            schema_version=config.observations.schema_version,
            schema_size=OBSERVATION_SIZE,
            schema_digest=OBSERVATION_SCHEMA_DIGEST,
            perception_radius=config.observations.perception_radius,
        ),
        genome=GenomeCoreConfig(
            schema_version=config.genome.schema_version,
            schema_digest=GENOME_SCHEMA_DIGEST,
            initialization_name=GENOME_INITIALIZATION_NAME,
            initialization_version=GENOME_INITIALIZATION_VERSION,
            parameter_count=GENOME_PARAMETER_COUNT,
        ),
        energy=EnergyCoreConfig(
            **{
                field: _float_scalar(value, f"energy.{field}")
                for field, value in config.energy.model_dump().items()
            }
        ),
        evolution=EvolutionCoreConfig(
            min_reproduction_age=_int_scalar(
                config.evolution.min_reproduction_age, "evolution.min_reproduction_age"
            ),
            max_age=_int_scalar(config.evolution.max_age, "evolution.max_age"),
            mutation_rate=_float_scalar(config.evolution.mutation_rate, "evolution.mutation_rate"),
            mutation_sigma=_float_scalar(
                config.evolution.mutation_sigma, "evolution.mutation_sigma"
            ),
            mutation_clip_abs=_float_scalar(
                config.evolution.mutation_clip_abs, "evolution.mutation_clip_abs"
            ),
        ),
        runtime=RuntimeCoreConfig(
            chunk_size=config.runtime.chunk_size,
            record_stride=config.runtime.record_stride,
            snapshot_stride=config.runtime.snapshot_stride,
        ),
    )
    return CompiledConfig(
        core=core,
        compile_signature=build_compile_signature(config),
        config_hash=freeze_config(config).config_hash,
    )
