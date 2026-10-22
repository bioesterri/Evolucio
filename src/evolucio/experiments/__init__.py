"""Explicit paired-seed experiments; importing plans never initializes JAX."""

from .batch import run_batch, seeds_for_case
from .preflight import BatchPreflight, CasePreflight, expand_batch_plan, preflight_batch
from .seeds import SeedSet
from .types import (
    BATCH_INTERFACE_VERSION,
    BatchDefinitionError,
    BatchExecutionError,
    BatchPlan,
    BatchResult,
    BatchRunResult,
    BatchRunSpec,
    BatchRunStatus,
    ConsumerFactory,
    ExperimentCase,
)

__all__ = [
    "BATCH_INTERFACE_VERSION",
    "BatchDefinitionError",
    "BatchExecutionError",
    "BatchPlan",
    "BatchPreflight",
    "BatchResult",
    "BatchRunResult",
    "BatchRunSpec",
    "BatchRunStatus",
    "CasePreflight",
    "ConsumerFactory",
    "ExperimentCase",
    "SeedSet",
    "expand_batch_plan",
    "preflight_batch",
    "run_batch",
    "seeds_for_case",
]
