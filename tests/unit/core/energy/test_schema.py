import ast
import re
from pathlib import Path

from evolucio.core.energy import (
    ENERGY_ACCOUNTING_SCHEMA_DIGEST,
    ENERGY_ACCOUNTING_SCHEMA_VERSION,
    energy_accounting_schema_digest,
    energy_accounting_schema_payload,
)


def test_schema_is_complete_stable_and_has_no_rng_dependency() -> None:
    payload = energy_accounting_schema_payload()
    assert ENERGY_ACCOUNTING_SCHEMA_VERSION == 1
    assert payload["rng"] == "none"
    assert energy_accounting_schema_digest() == ENERGY_ACCOUNTING_SCHEMA_DIGEST
    assert ENERGY_ACCOUNTING_SCHEMA_DIGEST == (
        "e880b08ada85e35196b2d5801d3f3104647a6980f692b84076bdcca57733f9b1"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", ENERGY_ACCOUNTING_SCHEMA_DIGEST)
    for path in Path("src/evolucio/core/energy").glob("*.py"):
        tree = ast.parse(path.read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert "evolucio.core.rng" not in imports
