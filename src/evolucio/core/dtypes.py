"""Canonical dtypes for arrays owned by the simulation core."""

import jax.numpy as jnp

REAL_DTYPE = jnp.float32
INDEX_DTYPE = jnp.int32
ID_DTYPE = jnp.int32
COUNT_DTYPE = jnp.int32
STEP_DTYPE = jnp.int32
CODE_DTYPE = jnp.int16
MASK_DTYPE = jnp.bool_
