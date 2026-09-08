import ast
import json
from pathlib import Path

from evolucio.config import CONFIG_SCHEMA_VERSION, experiment_config_json_schema


def test_schema_snapshot() -> None:
    schema = experiment_config_json_schema()
    assert CONFIG_SCHEMA_VERSION == "2.1"
    assert set(schema["required"]) >= {
        "world",
        "population",
        "policy",
        "energy",
        "evolution",
        "runtime",
    }
    assert schema["additionalProperties"] is False
    assert json.loads(Path("docs/schemas/experiment-config-v2.1.json").read_text()) == schema


def test_previous_schema_snapshot_remains_published() -> None:
    """Publishing 2.1 must not reinterpret the auditable 2.0 contract."""
    previous = json.loads(Path("docs/schemas/experiment-config-v2.0.json").read_text())

    assert previous["properties"]["schema_version"]["const"] == "2.0"
    assert previous["$defs"]["EnergyConfig"]["properties"]["reproduction_cost"]["minimum"] == 0


def test_config_has_no_forbidden_imports() -> None:
    forbidden = {
        "jax",
        "equinox",
        "evolucio.core",
        "evolucio.persistence",
        "evolucio.analysis",
        "evolucio.visualization",
    }
    for path in Path("src/evolucio/config").glob("*.py"):
        if path.name in {"compile.py", "__init__.py"}:
            continue
        tree = ast.parse(path.read_text())
        names = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        names += [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not any(
            any(name == item or name.startswith(item + ".") for item in forbidden) for name in names
        )
