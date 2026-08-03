"""Small host-side type aliases shared by core modules."""

import jax

type Array = jax.Array
type Shape = tuple[int, ...]
type AgentId = int
type GenomeId = int
type LineageId = int
type StepIndex = int
