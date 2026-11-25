"""Recovery preserves the original horizon and scientific configuration."""

from dataclasses import dataclass

from .errors import RecoveryPlanError


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    source_run_id: str
    checkpoint_step: int
    checkpoint_hash: str
    target_step: int
    remaining_steps: int
    config_hash: str
    chunk_steps: int
    recovery_schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not 0 <= self.checkpoint_step < self.target_step
            or self.remaining_steps != self.target_step - self.checkpoint_step
            or self.chunk_steps <= 0
        ):
            raise RecoveryPlanError("invalid recovery horizon or chunk length")
