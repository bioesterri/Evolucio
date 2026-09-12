# ruff: noqa: ANN201

import re

from evolucio.core.evolution import (
    REPRODUCTION_RESOLUTION_SCHEMA_DIGEST,
    REPRODUCTION_RESOLUTION_SCHEMA_NAME,
    REPRODUCTION_RESOLUTION_SCHEMA_VERSION,
    ReproductionResolutionCode,
    reproduction_resolution_schema_digest,
    reproduction_resolution_schema_payload,
)


def test_reproduction_codes_and_schema_are_frozen():
    assert [int(value) for value in ReproductionResolutionCode] == list(range(7))
    assert len(ReproductionResolutionCode.__members__) == 7
    assert REPRODUCTION_RESOLUTION_SCHEMA_NAME == "asexual_atomic_local_birth_fixed_slots_v1"
    assert REPRODUCTION_RESOLUTION_SCHEMA_VERSION == 1
    assert reproduction_resolution_schema_payload()["mutation"] == "none"
    assert reproduction_resolution_schema_digest() == REPRODUCTION_RESOLUTION_SCHEMA_DIGEST
    assert re.fullmatch(r"[0-9a-f]{64}", REPRODUCTION_RESOLUTION_SCHEMA_DIGEST)
    assert (
        REPRODUCTION_RESOLUTION_SCHEMA_DIGEST
        == "9510e8bb43f5783314406c5dd14474a0bd6fa7cd690ec8079ac245e3bd9e8108"
    )
