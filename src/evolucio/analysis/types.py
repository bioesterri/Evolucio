"""Versioned host contracts for persisted-run analysis."""

from dataclasses import dataclass
from enum import StrEnum

ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_SCHEMAS = {
    "population_analysis": 1,
    "energy_analysis": 1,
    "mortality_analysis": 1,
    "survival_analysis": 1,
}


def positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class AnalysisWindow:
    start_step: int
    end_step_exclusive: int

    def __post_init__(self) -> None:
        positive_int(self.end_step_exclusive, "end_step_exclusive")
        if type(self.start_step) is not int or not 0 <= self.start_step < self.end_step_exclusive:
            raise ValueError("expected 0 <= start_step < end_step_exclusive")


@dataclass(frozen=True, slots=True)
class AnalysisPolicy:
    window_steps: int = 100
    cohort_width_steps: int = 100
    bottleneck_fraction: float = 0.1

    def __post_init__(self) -> None:
        positive_int(self.window_steps, "window_steps")
        positive_int(self.cohort_width_steps, "cohort_width_steps")
        if isinstance(self.bottleneck_fraction, bool) or not 0 < self.bottleneck_fraction < 1:
            raise ValueError("bottleneck_fraction must be between zero and one")


class Quality(StrEnum):
    EXACT = "EXACT"
    SAMPLED = "SAMPLED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class Coverage:
    start_step: int
    end_step_exclusive: int


@dataclass(frozen=True, slots=True)
class Metric:
    value: float | None
    source: str
    coverage: Coverage
    quality: Quality
    limitations: str | None = None


class AnalysisUnavailableError(ValueError):
    """Required scientific evidence is unavailable."""


DEFAULT_ANALYSIS_POLICY = AnalysisPolicy()
