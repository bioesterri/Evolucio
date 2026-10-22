"""Host contracts for finite, streamed simulation runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from evolucio.core.metrics import MetricsAccumulator, StepEventBuffer, StepMetrics
    from evolucio.core.state import SimulationState


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Execution horizon and technical batch size, separate from CoreConfig."""

    total_steps: int
    chunk_steps: int

    def __post_init__(self) -> None:
        for name in ("total_steps", "chunk_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a Python int, not bool")
            if value <= 0:
                raise ValueError(f"{name} must be strictly positive")


@dataclass(frozen=True, slots=True)
class HostChunkBatch:
    """Complete time-stacked outputs after device_get, covering [start, end)."""

    chunk_index: int
    start_step: int
    end_step_exclusive: int
    metrics: StepMetrics
    events: StepEventBuffer


class ChunkConsumer(Protocol):
    """Synchronous host sink; exceptions abort the run and propagate."""

    def __call__(self, batch: HostChunkBatch) -> None:
        """Consume one complete batch without modifying the device carry."""
        ...


@dataclass(frozen=True, slots=True)
class RunRuntimeResult:
    """Final device carry and execution counts, without temporal history."""

    state: SimulationState
    accumulator: MetricsAccumulator
    initial_step: int
    final_step: int
    steps_executed: int
    chunks_executed: int
