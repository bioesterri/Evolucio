import json
import re

from evolucio.core.observations import (
    OBSERVATION_SCHEMA_DIGEST,
    OBSERVATION_SCHEMA_NAME,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SIZE,
    BlockedDirectionBit,
    ObservationIndex,
    observation_schema_digest,
    observation_schema_payload,
)
from evolucio.core.observations.schema import OBSERVATION_FIELDS


def test_frozen_schema_contract() -> None:
    assert OBSERVATION_SCHEMA_NAME == "local_cardinal_v1"
    assert OBSERVATION_SCHEMA_VERSION == 1
    assert OBSERVATION_SIZE == len(ObservationIndex) == len(OBSERVATION_FIELDS) == 15
    assert [member.value for member in ObservationIndex] == list(range(15))
    assert [field.index for field in OBSERVATION_FIELDS] == list(range(15))
    assert [field.name for field in OBSERVATION_FIELDS] == [
        member.name for member in ObservationIndex
    ]
    assert len(ObservationIndex.__members__) == 15


def test_schema_digest_and_payload_are_canonical() -> None:
    payload = observation_schema_payload()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["cardinal_rays"]["offsets"] == [[0, -1], [0, 1], [1, 0], [-1, 0]]
    assert payload["blocked_bits"] == {
        "north": 1,
        "south": 2,
        "east": 4,
        "west": 8,
        "normalizer": 15,
    }
    assert "map[y,x]" in payload["coordinates"]
    assert observation_schema_digest() == OBSERVATION_SCHEMA_DIGEST
    assert re.fullmatch("[0-9a-f]{64}", OBSERVATION_SCHEMA_DIGEST)


def test_blocked_direction_bits_are_stable() -> None:
    assert [bit.value for bit in BlockedDirectionBit] == [1, 2, 4, 8]
    assert len(BlockedDirectionBit.__members__) == 4
