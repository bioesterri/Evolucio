"""Compile validated host configuration into the JAX-facing contract."""

import dataclasses
import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from .freeze import canonical_json_and_hash, freeze_config
from .models import ExperimentConfig

COMPILE_SIGNATURE_SCHEMA_VERSION = 1
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
    environment_schedule_length: int
    max_agents: int
    max_births_per_step: int
    placement: str
    allow_multiple_agents_per_cell: bool
    observation_schema_version: str
    action_schema_version: str
    hidden_size: int
    activation: str
    perception_radius: int
    chunk_size: int
    record_stride: int
    snapshot_stride: int | None
    backend: str


class WorldCoreConfig(eqx.Module):
    """World parameters used by compiled core functions."""

    width: int
    height: int
    boundary_mode: str
    resource_distribution: str
    resource_capacity: jax.Array
    initial_resource_fraction: jax.Array
    regeneration_rate: jax.Array
    environment_start_steps: jax.Array
    environment_end_steps: jax.Array
    environment_regeneration_multipliers: jax.Array
    environment_stress_levels: jax.Array


class PopulationCoreConfig(eqx.Module):
    """Population parameters used by compiled core functions."""

    initial_agents: jax.Array
    max_agents: int
    max_births_per_step: int
    placement: str
    allow_multiple_agents_per_cell: bool


class PolicyCoreConfig(eqx.Module):
    """Fixed policy topology and schema selectors."""

    observation_schema_version: str
    action_schema_version: str
    hidden_size: int
    activation: str
    perception_radius: int


class EnergyCoreConfig(eqx.Module):
    """Dynamic energy economy parameters."""

    initial_energy: jax.Array
    max_energy: jax.Array
    death_threshold: jax.Array
    basal_cost: jax.Array
    movement_cost: jax.Array
    feeding_cost: jax.Array
    feeding_conversion: jax.Array
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
    result = jnp.asarray(value, dtype=jnp.int32)  # pyright: ignore[reportUnknownMemberType]
    if result.shape != ():
        raise ConfigCompilationError(f"{field} must compile to a scalar")
    return result


def _float_scalar(value: float, field: str) -> jax.Array:
    if not math.isfinite(value):
        raise ConfigCompilationError(f"{field} must be finite")
    if abs(value) > _FLOAT32_MAX:
        raise ConfigCompilationError(f"{field} is outside the supported float32 range")
    result = jnp.asarray(value, dtype=jnp.float32)  # pyright: ignore[reportUnknownMemberType]
    if result.shape != () or not math.isfinite(float(result)):
        raise ConfigCompilationError(f"{field} must compile to a finite scalar")
    if value != 0.0 and float(result) == 0.0:
        raise ConfigCompilationError(f"{field} underflows to zero in float32")
    return result


def _int_vector(values: tuple[int, ...], field: str) -> jax.Array:
    for value in values:
        _static_int(value, field)
    return jnp.asarray(values, dtype=jnp.int32)  # pyright: ignore[reportUnknownMemberType]


def _float_vector(values: tuple[float, ...], field: str) -> jax.Array:
    for index, value in enumerate(values):
        _float_scalar(value, f"{field}[{index}]")
    result = jnp.asarray(values, dtype=jnp.float32)  # pyright: ignore[reportUnknownMemberType]
    if not bool(jnp.all(jnp.isfinite(result))):
        raise ConfigCompilationError(f"{field} must contain only finite values")
    return result


def _validate_compiled_energy(energy: EnergyCoreConfig) -> None:
    """Recheck host energy invariants after independent float32 conversion."""
    death_threshold = float(energy.death_threshold)
    initial_energy = float(energy.initial_energy)
    max_energy = float(energy.max_energy)
    reproduction_threshold = float(energy.reproduction_threshold)
    reproduction_cost = float(energy.reproduction_cost)
    offspring_initial_energy = float(energy.offspring_initial_energy)

    if not death_threshold < initial_energy <= max_energy:
        raise ConfigCompilationError(
            "energy.initial_energy violates viability bounds after float32 conversion"
        )
    if max_energy <= death_threshold:
        raise ConfigCompilationError(
            "energy.max_energy is not above death_threshold after float32 conversion"
        )
    if reproduction_threshold > max_energy:
        raise ConfigCompilationError(
            "energy.reproduction_threshold exceeds max_energy after float32 conversion"
        )
    if offspring_initial_energy <= death_threshold:
        raise ConfigCompilationError(
            "energy.offspring_initial_energy is not viable after float32 conversion"
        )
    minimum = death_threshold + reproduction_cost + offspring_initial_energy
    if reproduction_threshold <= minimum:
        raise ConfigCompilationError(
            "energy.reproduction_threshold does not leave the parent viable after float32 "
            "conversion"
        )


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
        environment_schedule_length=_static_int(
            len(world.environment_schedule), "world.environment_schedule"
        ),
        max_agents=_static_int(population.max_agents, "population.max_agents"),
        max_births_per_step=_static_int(
            population.max_births_per_step, "population.max_births_per_step"
        ),
        placement=population.placement,
        allow_multiple_agents_per_cell=population.allow_multiple_agents_per_cell,
        observation_schema_version=policy.observation_schema_version,
        action_schema_version=policy.action_schema_version,
        hidden_size=_static_int(policy.hidden_size, "policy.hidden_size"),
        activation=policy.activation,
        perception_radius=_static_int(policy.perception_radius, "policy.perception_radius"),
        chunk_size=_static_int(runtime.chunk_size, "runtime.chunk_size"),
        record_stride=_static_int(runtime.record_stride, "runtime.record_stride"),
        snapshot_stride=(
            None
            if runtime.snapshot_stride is None
            else _static_int(runtime.snapshot_stride, "runtime.snapshot_stride")
        ),
        backend=runtime.backend,
    )


def compile_signature_digest(signature: CompileSignature) -> str:
    """Return a process-independent SHA-256 identity for a compile signature."""
    _, digest = canonical_json_and_hash(dataclasses.asdict(signature))
    return digest


def compile_config(config: ExperimentConfig) -> CompiledConfig:
    """Compile an already validated host model without I/O or side effects."""
    world = config.world
    phases = world.environment_schedule
    energy = EnergyCoreConfig(
        initial_energy=_float_scalar(config.energy.initial_energy, "energy.initial_energy"),
        max_energy=_float_scalar(config.energy.max_energy, "energy.max_energy"),
        death_threshold=_float_scalar(config.energy.death_threshold, "energy.death_threshold"),
        basal_cost=_float_scalar(config.energy.basal_cost, "energy.basal_cost"),
        movement_cost=_float_scalar(config.energy.movement_cost, "energy.movement_cost"),
        feeding_cost=_float_scalar(config.energy.feeding_cost, "energy.feeding_cost"),
        feeding_conversion=_float_scalar(
            config.energy.feeding_conversion, "energy.feeding_conversion"
        ),
        reproduction_threshold=_float_scalar(
            config.energy.reproduction_threshold, "energy.reproduction_threshold"
        ),
        reproduction_cost=_float_scalar(
            config.energy.reproduction_cost, "energy.reproduction_cost"
        ),
        offspring_initial_energy=_float_scalar(
            config.energy.offspring_initial_energy, "energy.offspring_initial_energy"
        ),
        failed_action_cost=_float_scalar(
            config.energy.failed_action_cost, "energy.failed_action_cost"
        ),
    )
    _validate_compiled_energy(energy)
    core = CoreConfig(
        world=WorldCoreConfig(
            width=_static_int(world.width, "world.width"),
            height=_static_int(world.height, "world.height"),
            boundary_mode=world.boundary_mode,
            resource_distribution=world.resource_distribution,
            resource_capacity=_float_scalar(world.resource_capacity, "world.resource_capacity"),
            initial_resource_fraction=_float_scalar(
                world.initial_resource_fraction, "world.initial_resource_fraction"
            ),
            regeneration_rate=_float_scalar(world.regeneration_rate, "world.regeneration_rate"),
            environment_start_steps=_int_vector(
                tuple(phase.start_step for phase in phases), "world.environment_schedule.start_step"
            ),
            environment_end_steps=_int_vector(
                tuple(phase.end_step for phase in phases), "world.environment_schedule.end_step"
            ),
            environment_regeneration_multipliers=_float_vector(
                tuple(phase.regeneration_multiplier for phase in phases),
                "world.environment_schedule.regeneration_multiplier",
            ),
            environment_stress_levels=_float_vector(
                tuple(phase.stress_level for phase in phases),
                "world.environment_schedule.stress_level",
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
        policy=PolicyCoreConfig(**config.policy.model_dump()),
        energy=energy,
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
