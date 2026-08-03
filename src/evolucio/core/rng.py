"""Functional deterministic random-key management."""

import equinox as eqx
import jax

from .codes import RngStreamCode
from .dtypes import ID_DTYPE
from .types import Array

PRNG_IMPLEMENTATION = "threefry2x32"
_MIN_SEED = 0
_MAX_SEED = 2**32 - 1


class RngState(eqx.Module):
    """The single persistent typed root key."""

    key: Array


def create_rng_state(seed: int) -> RngState:
    """Create a typed root key from a validated host seed."""
    if isinstance(seed, bool) or not isinstance(seed, int):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("seed must be a Python int, not bool")
    if not _MIN_SEED <= seed <= _MAX_SEED:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    return RngState(key=jax.random.key(seed, impl=PRNG_IMPLEMENTATION))


def advance_rng(rng: RngState) -> tuple[RngState, Array]:
    """Split once, returning the first child as root and the second as step key."""
    next_key, step_key = jax.random.split(rng.key, num=2)
    return RngState(key=next_key), step_key


def derive_stream_key(step_key: Array, stream: RngStreamCode) -> Array:
    """Derive a stable domain key without advancing the root key."""
    if not isinstance(stream, RngStreamCode):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("stream must be a RngStreamCode")
    return jax.random.fold_in(step_key, int(stream))


def derive_indexed_key(base_key: Array, index: Array | int) -> Array:
    """Derive a child key from one explicit scalar identity or index."""
    if isinstance(index, bool):
        raise TypeError("index must be an integer scalar, not bool")
    if isinstance(index, int):
        if not -(2**31) <= index <= 2**31 - 1:
            raise ValueError("index must be representable as int32")
        value: Array | int = index
    else:
        if index.shape != ():
            raise ValueError("index must be scalar")
        if index.dtype != ID_DTYPE:
            raise TypeError("index array must use ID_DTYPE")
        value = index
    return jax.random.fold_in(base_key, value)


def derive_entity_keys(base_key: Array, entity_ids: Array) -> Array:
    """Vectorise identity-based child-key derivation over entity IDs."""
    if entity_ids.ndim != 1:
        raise ValueError("entity_ids must be one-dimensional")
    if entity_ids.dtype != ID_DTYPE:
        raise TypeError("entity_ids must use ID_DTYPE")
    return jax.vmap(jax.random.fold_in, in_axes=(None, 0))(base_key, entity_ids)
