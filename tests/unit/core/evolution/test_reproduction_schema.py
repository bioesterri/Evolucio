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
    assert REPRODUCTION_RESOLUTION_SCHEMA_NAME == "asexual_atomic_local_birth_fixed_slots_v2"
    assert REPRODUCTION_RESOLUTION_SCHEMA_VERSION == 2
    assert reproduction_resolution_schema_payload()["birth_count_limit"] == (
        "at_most_configured_max_births_per_step"
    )
    assert reproduction_resolution_schema_payload()["mutation"] == "none"
    assert reproduction_resolution_schema_digest() == REPRODUCTION_RESOLUTION_SCHEMA_DIGEST
    assert re.fullmatch(r"[0-9a-f]{64}", REPRODUCTION_RESOLUTION_SCHEMA_DIGEST)
    assert (
        REPRODUCTION_RESOLUTION_SCHEMA_DIGEST
        == "33df5c1f6e7a21743ef616fbd7580b7e018014c46612077eb9578b59e0b4256c"
    )
