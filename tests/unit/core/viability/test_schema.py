# ruff: noqa: ANN201
import re

from evolucio.core.viability import (
    PRE_ACTION_VIABILITY_SCHEMA_DIGEST,
    PRE_ACTION_VIABILITY_SCHEMA_NAME,
    PRE_ACTION_VIABILITY_SCHEMA_VERSION,
    pre_action_viability_schema_digest,
    pre_action_viability_schema_payload,
)


def test_schema_is_frozen():
    payload = pre_action_viability_schema_payload()
    assert PRE_ACTION_VIABILITY_SCHEMA_NAME == "invalid_energy_age_preaction_death_v1"
    assert PRE_ACTION_VIABILITY_SCHEMA_VERSION == 1
    assert payload["cause_priority"] == ["INVALID_STATE", "ENERGY_DEPLETION", "MAX_AGE"]
    assert payload["direct_environmental_death"] == payload["direct_competitive_death"] == "none"
    assert pre_action_viability_schema_digest() == PRE_ACTION_VIABILITY_SCHEMA_DIGEST
    assert (
        PRE_ACTION_VIABILITY_SCHEMA_DIGEST
        == "64ca5bf100fe1a8cb1f9cf6ad286ddc45e2c7615eddb8f24f49a813351f176a6"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", PRE_ACTION_VIABILITY_SCHEMA_DIGEST)
