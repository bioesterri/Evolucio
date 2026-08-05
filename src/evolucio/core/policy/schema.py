"""Persistent contract for the fixed single-agent policy."""

import hashlib
import json
from dataclasses import asdict, dataclass

from evolucio.core.codes import ACTION_COUNT, ActionCode
from evolucio.core.observations import (
    OBSERVATION_SCHEMA_NAME,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SIZE,
    observation_schema_digest,
)

POLICY_SCHEMA_NAME = "policy_mlp_tanh_15_16_7_v1"
POLICY_SCHEMA_VERSION = 1
POLICY_INPUT_SIZE = OBSERVATION_SIZE
POLICY_HIDDEN_SIZE = 16
POLICY_OUTPUT_SIZE = ACTION_COUNT
POLICY_ACTIVATION_NAME = "tanh"
POLICY_USE_BIAS = True


@dataclass(frozen=True, slots=True)
class PolicyParameterSpec:
    """One stable neural parameter leaf in serialization order."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    count: int


POLICY_PARAMETER_SPECS = (
    PolicyParameterSpec(
        "layer1.weight",
        (POLICY_HIDDEN_SIZE, POLICY_INPUT_SIZE),
        "float32",
        POLICY_HIDDEN_SIZE * POLICY_INPUT_SIZE,
    ),
    PolicyParameterSpec("layer1.bias", (POLICY_HIDDEN_SIZE,), "float32", POLICY_HIDDEN_SIZE),
    PolicyParameterSpec(
        "layer2.weight",
        (POLICY_OUTPUT_SIZE, POLICY_HIDDEN_SIZE),
        "float32",
        POLICY_OUTPUT_SIZE * POLICY_HIDDEN_SIZE,
    ),
    PolicyParameterSpec("layer2.bias", (POLICY_OUTPUT_SIZE,), "float32", POLICY_OUTPUT_SIZE),
)
POLICY_PARAMETER_COUNT = sum(spec.count for spec in POLICY_PARAMETER_SPECS)


def policy_schema_payload() -> dict[str, object]:
    """Return the complete JSON-compatible topology descriptor."""
    return {
        "name": POLICY_SCHEMA_NAME,
        "version": POLICY_SCHEMA_VERSION,
        "model_type": "PolicyMLP",
        "input_size": POLICY_INPUT_SIZE,
        "hidden_size": POLICY_HIDDEN_SIZE,
        "output_size": POLICY_OUTPUT_SIZE,
        "hidden_layer_count": 1,
        "hidden_activation": POLICY_ACTIVATION_NAME,
        "output_activation": None,
        "use_bias": POLICY_USE_BIAS,
        "dtype": "float32",
        "parameters": [asdict(spec) for spec in POLICY_PARAMETER_SPECS],
        "parameter_count": POLICY_PARAMETER_COUNT,
        "observation_schema": {
            "name": OBSERVATION_SCHEMA_NAME,
            "version": OBSERVATION_SCHEMA_VERSION,
            "size": OBSERVATION_SIZE,
            "digest": observation_schema_digest(),
        },
        "action_codes": [{"name": code.name, "value": int(code)} for code in ActionCode],
        "output_semantics": "linear_action_scores",
    }


def policy_schema_digest() -> str:
    """Return the canonical SHA-256 digest of the policy schema."""
    canonical = json.dumps(
        policy_schema_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


POLICY_SCHEMA_DIGEST = policy_schema_digest()
