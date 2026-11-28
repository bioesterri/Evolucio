"""Paired multi-seed comparison of verified experimental trajectories."""

from .compare import ExperimentComparisonResult, compare_experiment
from .export import export_comparison
from .selection import LocalRunProvider, RegisteredRunProvider, select_comparison_runs
from .statistics import (
    BootstrapPolicy,
    bootstrap_paired_mean,
    compare_paired_metrics,
    summarize_case_metrics,
)
from .temporal import AlignmentPolicy, compare_time_series

__all__ = [
    "AlignmentPolicy",
    "BootstrapPolicy",
    "ExperimentComparisonResult",
    "LocalRunProvider",
    "RegisteredRunProvider",
    "bootstrap_paired_mean",
    "compare_experiment",
    "compare_paired_metrics",
    "compare_time_series",
    "export_comparison",
    "select_comparison_runs",
    "summarize_case_metrics",
]
