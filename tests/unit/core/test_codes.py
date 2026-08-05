import json

import jax.numpy as jnp

from evolucio.core import (
    ACTION_COUNT,
    CODE_DTYPE,
    DEATH_CAUSE_COUNT,
    RNG_STREAM_COUNT,
    ActionCode,
    DeathCauseCode,
    RngStreamCode,
)


def test_action_codes_are_stable_and_complete() -> None:
    expected = {
        "STAY": 0,
        "MOVE_NORTH": 1,
        "MOVE_SOUTH": 2,
        "MOVE_EAST": 3,
        "MOVE_WEST": 4,
        "EAT": 5,
        "REPRODUCE": 6,
    }
    assert {code.name: int(code) for code in ActionCode} == expected
    assert [int(code) for code in ActionCode] == list(range(ACTION_COUNT))
    assert ACTION_COUNT == 7


def test_action_codes_round_trip_through_jax_integer_arrays() -> None:
    values = jnp.asarray([int(code) for code in ActionCode], dtype=CODE_DTYPE)
    result = values + jnp.asarray(0, dtype=CODE_DTYPE)
    assert result.dtype == jnp.dtype(CODE_DTYPE)
    assert result.tolist() == list(range(ACTION_COUNT))


def test_death_cause_codes_are_stable_and_complete() -> None:
    expected = {
        "NONE": 0,
        "ENERGY_DEPLETION": 1,
        "MAX_AGE": 2,
        "ENVIRONMENTAL_STRESS": 3,
        "COMPETITIVE_EXCLUSION": 4,
        "INVALID_STATE": 5,
    }
    assert {code.name: int(code) for code in DeathCauseCode} == expected
    assert [int(code) for code in DeathCauseCode] == list(range(DEATH_CAUSE_COUNT))
    assert DEATH_CAUSE_COUNT == 6
    assert int(DeathCauseCode.NONE) == 0


def test_death_cause_codes_round_trip_through_jax_integer_arrays() -> None:
    values = jnp.asarray([int(code) for code in DeathCauseCode], dtype=CODE_DTYPE)
    assert values.dtype == jnp.dtype(CODE_DTYPE)
    assert values.tolist() == list(range(DEATH_CAUSE_COUNT))


def test_codes_serialize_as_their_stable_integer_values() -> None:
    assert json.dumps(ActionCode.EAT) == "5"
    assert json.dumps(DeathCauseCode.INVALID_STATE) == "5"


def test_rng_stream_codes_are_stable_and_complete() -> None:
    expected = {
        "WORLD_INITIALIZATION": 0,
        "RESOURCE_INITIALIZATION": 1,
        "AGENT_INITIALIZATION": 2,
        "GENOME_INITIALIZATION": 3,
        "ENVIRONMENT_UPDATE": 4,
        "ACTION_TIE_BREAK": 5,
        "MOVEMENT_CONFLICT": 6,
        "RESOURCE_CONFLICT": 7,
        "REPRODUCTION_CONFLICT": 8,
        "BIRTH_PLACEMENT": 9,
        "GENOME_MUTATION": 10,
    }
    assert {code.name: int(code) for code in RngStreamCode} == expected
    assert [int(code) for code in RngStreamCode] == list(range(RNG_STREAM_COUNT))
    assert RNG_STREAM_COUNT == 11


def test_rng_stream_codes_round_trip_through_jax_integer_arrays() -> None:
    values = jnp.asarray([int(code) for code in RngStreamCode], dtype=CODE_DTYPE)
    assert values.dtype == jnp.dtype(CODE_DTYPE)
    assert values.tolist() == list(range(RNG_STREAM_COUNT))
