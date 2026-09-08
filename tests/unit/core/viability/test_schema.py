# ruff: noqa: ANN201
import re

from evolucio.core.viability import (
    POST_ACTION_VIABILITY_SCHEMA_DIGEST,
    POST_ACTION_VIABILITY_SCHEMA_NAME,
    POST_ACTION_VIABILITY_SCHEMA_VERSION,
    PRE_ACTION_VIABILITY_SCHEMA_DIGEST,
    PRE_ACTION_VIABILITY_SCHEMA_NAME,
    PRE_ACTION_VIABILITY_SCHEMA_VERSION,
    post_action_viability_schema_digest,
    post_action_viability_schema_payload,
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


def test_post_action_schema_is_frozen():
    payload = post_action_viability_schema_payload()
    assert POST_ACTION_VIABILITY_SCHEMA_NAME == (
        "postaction_death_and_projected_reproduction_survival_v1"
    )
    assert POST_ACTION_VIABILITY_SCHEMA_VERSION == 1
    assert payload["cause_priority"] == ["INVALID_STATE", "ENERGY_DEPLETION", "MAX_AGE"]
    assert payload["death_before_reproduction"] is True
    assert payload["projected_parent_energy"] == (
        "energy - reproduction_energy_cost - offspring_initial_energy"
    )
    assert payload["projected_survival"] == ("projected_parent_energy > death_energy_threshold")
    assert payload["spatial_availability_check"] == payload["rng"] == "none"
    assert post_action_viability_schema_digest() == POST_ACTION_VIABILITY_SCHEMA_DIGEST
    assert (
        POST_ACTION_VIABILITY_SCHEMA_DIGEST
        == "2125e8425d9c397ee85f58cadf416d9135de125d3bfc10c71c7ebc6e6e38c08e"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", POST_ACTION_VIABILITY_SCHEMA_DIGEST)
