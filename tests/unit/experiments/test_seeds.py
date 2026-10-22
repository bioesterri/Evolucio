# ruff: noqa: ANN001, ANN201

from dataclasses import FrozenInstanceError

import pytest

from evolucio.experiments import SeedSet


@pytest.mark.parametrize("values", [(11,), (33, 11, 22), (0, 2**32 - 1)])
def test_preserved_immutable_seeds(values):
    seeds = SeedSet(values)
    assert seeds.seeds == values
    with pytest.raises(FrozenInstanceError):
        seeds.seeds = (1,)


@pytest.mark.parametrize("values", [(), (1, 2, 2, 3), (-1,), (2**32,)])
def test_invalid_values(values):
    with pytest.raises(ValueError):
        SeedSet(values)


@pytest.mark.parametrize("values", [(True,), (False,), (1.0,), ("1",), [1, 2]])
def test_invalid_types(values):
    with pytest.raises(TypeError):
        SeedSet(values)
