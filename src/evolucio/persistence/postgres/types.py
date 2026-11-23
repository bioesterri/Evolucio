"""Host-side value objects for the PostgreSQL registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from .models import GenomeHighlightModel, GenomeRegistryModel


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: UUID
    name: str
    description: str | None
    objective: str | None
    hypothesis: str | None
    label: str | None
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RunSummaryRecord:
    alive_count: int
    births_total: int
    deaths_total: int
    active_lineage_count: int
    final_total_energy: float
    final_total_resources: float

    @property
    def population_extinct(self) -> bool:
        return self.alive_count == 0


@dataclass(frozen=True, slots=True)
class ComparisonRun:
    run_id: UUID
    case_id: str | None
    config_hash: str
    seed: int
    status: str
    mode: str
    summary: RunSummaryRecord | None


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    ok: bool
    missing_in_database: tuple[str, ...]
    missing_locally: tuple[str, ...]
    hash_mismatches: tuple[str, ...]
    metadata_mismatches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenomeHighlights:
    genome: GenomeRegistryModel
    highlights: tuple[GenomeHighlightModel, ...]
