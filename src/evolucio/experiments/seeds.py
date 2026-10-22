"""Explicit, ordered master seeds for paired experimental runs."""

# pyright: reportUnnecessaryIsInstance=false

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeedSet:
    """Unique Python integer seeds in the canonical uint32 RNG range."""

    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.seeds, tuple):
            raise TypeError("seeds must be an immutable tuple")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        for seed in self.seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("seed must be a Python int, not bool")
            # Same host range as core.rng.create_rng_state; no device import here.
            if not 0 <= seed <= 2**32 - 1:
                raise ValueError("seed must be between 0 and 2**32 - 1")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate seeds are not allowed")
