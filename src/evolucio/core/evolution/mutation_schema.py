"""Versioned contract for newborn neural-genome mutation."""

import hashlib
import json

GENOME_MUTATION_SCHEMA_NAME = "independent_gaussian_weight_bias_clipped_v1"
GENOME_MUTATION_SCHEMA_VERSION = 1
GENOME_MUTATION_DISTRIBUTION = "normal_additive"
GENOME_MUTATION_SCOPE = "newborns_only"
GENOME_MUTATION_MASK_POLICY = "independent_per_parameter"
GENOME_MUTATION_LIMIT_POLICY = "symmetric_clip"


def genome_mutation_schema_payload() -> dict[str, object]:
    """Return the canonical semantic contract for genome mutation."""
    return {
        "schema_name": GENOME_MUTATION_SCHEMA_NAME,
        "schema_version": GENOME_MUTATION_SCHEMA_VERSION,
        "scope": GENOME_MUTATION_SCOPE,
        "heritable_leaves": {
            "weights": ["layer1.weight", "layer2.weight"],
            "biases": ["layer1.bias", "layer2.bias"],
        },
        "mutation": {
            "mask_distribution": "bernoulli",
            "mask_policy": GENOME_MUTATION_MASK_POLICY,
            "distribution": GENOME_MUTATION_DISTRIBUTION,
            "noise_scale": "sigma",
            "operation": "inherited_value + selected * noise",
            "limit_policy": GENOME_MUTATION_LIMIT_POLICY,
            "limits": "group_specific_positive_abs_limit",
            "structural_mutations": False,
        },
        "rng": {
            "stream": "GENOME_MUTATION",
            "identity": "child_genome_id",
            "slot_independent": True,
            "substreams": "independent_leaf_mask_and_noise",
        },
        "summary": {
            "per_slot": {
                "selected_parameter_count": "bernoulli-selected elements",
                "effective_parameter_count": "elements changed after clipping",
                "selected_weight_count": "selected weight elements",
                "selected_bias_count": "selected bias elements",
                "sum_abs_delta": "sum absolute post-clipping delta",
                "max_abs_delta": "maximum absolute post-clipping delta",
            },
            "scalar": {
                "mutated_newborn_count": "valid newborns with an effective change",
                "invalid_newborn_genome_count": "newborn inputs rejected defensively",
            },
        },
    }


def genome_mutation_schema_digest() -> str:
    """Return the SHA-256 digest of canonical compact JSON."""
    canonical = json.dumps(
        genome_mutation_schema_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


GENOME_MUTATION_SCHEMA_DIGEST = genome_mutation_schema_digest()
