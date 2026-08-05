import jax.numpy as jnp
import pytest

from evolucio.core import (
    CODE_DTYPE,
    COUNT_DTYPE,
    ID_DTYPE,
    INDEX_DTYPE,
    MASK_DTYPE,
    REAL_DTYPE,
    STEP_DTYPE,
)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (REAL_DTYPE, jnp.float32),
        (INDEX_DTYPE, jnp.int32),
        (ID_DTYPE, jnp.int32),
        (COUNT_DTYPE, jnp.int32),
        (STEP_DTYPE, jnp.int32),
        (CODE_DTYPE, jnp.int16),
        (MASK_DTYPE, jnp.bool_),
    ],
)
def test_core_dtype_policy(dtype: object, expected: object) -> None:
    assert dtype is expected
    array = jnp.asarray([0, 1], dtype=dtype)
    assert array.dtype == jnp.dtype(expected)
    assert array.dtype not in {jnp.dtype("float64"), jnp.dtype("int64")}
