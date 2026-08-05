"""Stable host configuration API."""

from .compile import (
    COMPILE_SIGNATURE_SCHEMA_VERSION,
    CompiledConfig,
    CompileSignature,
    ConfigCompilationError,
    CoreConfig,
    EnergyCoreConfig,
    EnvironmentCalendarCoreConfig,
    EvolutionCoreConfig,
    PolicyCoreConfig,
    PopulationCoreConfig,
    RuntimeCoreConfig,
    WorldCoreConfig,
    build_compile_signature,
    compile_config,
    compile_signature_digest,
)
from .freeze import FrozenConfig, freeze_config
from .io import ConfigFormat, dump_config, load_config, parse_config, serialize_config
from .models import (
    EnergyConfig,
    EnvironmentPhaseConfig,
    EvolutionConfig,
    ExperimentConfig,
    GenomeConfig,
    ObservationsConfig,
    PersistenceConfig,
    PolicyConfig,
    PopulationConfig,
    RuntimeConfig,
    WorldConfig,
)
from .schema import experiment_config_json_schema
from .versions import CONFIG_SCHEMA_VERSION

_COMPILE_EXPORTS = frozenset(
    {
        "COMPILE_SIGNATURE_SCHEMA_VERSION",
        "CompiledConfig",
        "CompileSignature",
        "ConfigCompilationError",
        "CoreConfig",
        "EnergyCoreConfig",
        "EnvironmentCalendarCoreConfig",
        "EvolutionCoreConfig",
        "GenomeCoreConfig",
        "ObservationsCoreConfig",
        "PolicyCoreConfig",
        "PopulationCoreConfig",
        "RuntimeCoreConfig",
        "WorldCoreConfig",
        "build_compile_signature",
        "compile_config",
        "compile_signature_digest",
    }
)


def __getattr__(name: str) -> object:
    """Load the JAX-facing configuration API only when explicitly requested."""
    if name in _COMPILE_EXPORTS:
        module = import_module(".compile", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "COMPILE_SIGNATURE_SCHEMA_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "CompileSignature",
    "CompiledConfig",
    "ConfigCompilationError",
    "ConfigFormat",
    "CoreConfig",
    "EnergyConfig",
    "EnergyCoreConfig",
    "EnvironmentCalendarCoreConfig",
    "EnvironmentPhaseConfig",
    "EvolutionConfig",
    "EvolutionCoreConfig",
    "ExperimentConfig",
    "FrozenConfig",
    "GenomeConfig",
    "GenomeCoreConfig",
    "ObservationsConfig",
    "ObservationsCoreConfig",
    "PersistenceConfig",
    "PolicyConfig",
    "PolicyCoreConfig",
    "PopulationConfig",
    "PopulationCoreConfig",
    "RuntimeConfig",
    "RuntimeCoreConfig",
    "WorldConfig",
    "WorldCoreConfig",
    "build_compile_signature",
    "compile_config",
    "compile_signature_digest",
    "dump_config",
    "experiment_config_json_schema",
    "freeze_config",
    "load_config",
    "parse_config",
    "serialize_config",
]
