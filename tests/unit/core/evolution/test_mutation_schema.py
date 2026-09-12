import re

from evolucio.core.evolution import (
    GENOME_MUTATION_SCHEMA_DIGEST,
    GENOME_MUTATION_SCHEMA_VERSION,
    GenomeMutationStreamCode,
    genome_mutation_schema_digest,
    genome_mutation_schema_payload,
)


def test_schema_payload_and_frozen_digest() -> None:
    payload = genome_mutation_schema_payload()
    assert GENOME_MUTATION_SCHEMA_VERSION == 1
    assert payload["scope"] == "newborns_only"
    assert payload["heritable_leaves"] == {
        "weights": ["layer1.weight", "layer2.weight"],
        "biases": ["layer1.bias", "layer2.bias"],
    }
    assert genome_mutation_schema_digest() == GENOME_MUTATION_SCHEMA_DIGEST
    assert (
        GENOME_MUTATION_SCHEMA_DIGEST
        == "891f2946784dedac15edb53eacf0c0816c3c05e7da83aabfd745c681648050be"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", GENOME_MUTATION_SCHEMA_DIGEST)


def test_all_leaf_mask_and_noise_substreams_are_explicit_and_unique() -> None:
    assert [int(code) for code in GenomeMutationStreamCode] == list(range(8))
