"""Validation of materialized host arrays, never implicit device transfers."""

from typing import cast

import numpy as np

from evolucio.host.dto import HostArray


def host_array(value: object, name: str, kind: str, shape: tuple[int, ...]) -> HostArray:
    """Reject device data, malformed shapes and incompatible scalar dtypes."""
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a materialized NumPy host array")
    array = cast(HostArray, value)
    if array.shape != shape or array.dtype.kind not in kind:
        raise ValueError(f"{name} has invalid shape or dtype")
    return array


def finite(array: HostArray, name: str) -> None:
    """Reject non-finite data without clipping or repair."""
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} contains non-finite values")


def nonnegative(array: HostArray, name: str) -> None:
    """Validate nonnegative coordinates, counts and required IDs."""
    if bool(np.any(array < 0)):
        raise ValueError(f"{name} must be nonnegative")


def host_ndim(value: object, name: str, kind: str, ndim: int) -> HostArray:
    """Validate host materialization before inspecting dimensions or values."""
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a materialized NumPy host array")
    array = cast(HostArray, value)
    if array.ndim != ndim:
        raise ValueError(f"{name} has invalid dimensions")
    return host_array(array, name, kind, array.shape)
