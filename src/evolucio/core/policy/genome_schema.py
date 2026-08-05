"""Canonical schema for the v1 batched neural genome."""

import hashlib
import json
from dataclasses import asdict

from evolucio.core.codes import ActionCode
from evolucio.core.observations import (
    OBSERVATION_SCHEMA_NAME,
    OBSERVATION_SCHEMA_VERSION,
    observation_schema_digest,
)

from .schema import (
    POLICY_PARAMETER_COUNT,
    POLICY_PARAMETER_SPECS,
    POLICY_SCHEMA_NAME,
    POLICY_SCHEMA_VERSION,
    policy_schema_digest,
)

GENOME_SCHEMA_NAME = "neural_genome_policy_mlp_15_16_7_v1"
GENOME_SCHEMA_VERSION = 1
GENOME_PARAMETER_COUNT = POLICY_PARAMETER_COUNT
GENOME_INITIALIZATION_NAME = "glorot_uniform_zero_bias_v1"
GENOME_INITIALIZATION_VERSION = 1


def genome_schema_payload() -> dict[str, object]:
    """Return the complete JSON-compatible individual-genome descriptor."""
    return {
        "name": GENOME_SCHEMA_NAME,
        "version": GENOME_SCHEMA_VERSION,
        "policy_schema": {
            "name": POLICY_SCHEMA_NAME,
            "version": POLICY_SCHEMA_VERSION,
            "digest": policy_schema_digest(),
        },
        "observation_schema": {
            "name": OBSERVATION_SCHEMA_NAME,
            "version": OBSERVATION_SCHEMA_VERSION,
            "digest": observation_schema_digest(),
        },
        "action_codes": [{"name": code.name, "value": int(code)} for code in ActionCode],
        "parameters": [
            {**asdict(spec), "batched_shape": ["C", *spec.shape]} for spec in POLICY_PARAMETER_SPECS
        ],
        "dtype": "float32",
        "parameter_count": GENOME_PARAMETER_COUNT,
        "inactive_slot_representation": "all_four_parameter_leaves_exactly_zero",
        "initialization": {
            "name": GENOME_INITIALIZATION_NAME,
            "version": GENOME_INITIALIZATION_VERSION,
            "weight_formula": "uniform(-sqrt(6/(fan_in+fan_out)),sqrt(6/(fan_in+fan_out)))",
            "bias": "zero",
            "substreams": ["LAYER1_WEIGHT=0", "LAYER2_WEIGHT=1"],
        },
        "identity_binding": "population.genome_id at the same fixed-capacity slot",
    }


def genome_schema_digest() -> str:
    """Return the canonical SHA-256 digest of the genome schema."""
    canonical = json.dumps(
        genome_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


GENOME_SCHEMA_DIGEST = genome_schema_digest()
