import ast
import json
import subprocess
import sys
from pathlib import Path

from evolucio.config import CONFIG_SCHEMA_VERSION, experiment_config_json_schema


def test_schema_snapshot() -> None:
    schema = experiment_config_json_schema()
    assert CONFIG_SCHEMA_VERSION == "1.6"
    assert set(schema["required"]) >= {
        "world",
        "population",
        "policy",
        "energy",
        "evolution",
        "runtime",
    }
    assert schema["additionalProperties"] is False
    assert json.loads(Path("docs/schemas/experiment-config-v1.6.json").read_text()) == schema


def test_previous_schema_snapshot_remains_published() -> None:
    """Publishing 1.6 must not remove or reinterpret the auditable 1.5 contract."""
    previous = json.loads(Path("docs/schemas/experiment-config-v1.5.json").read_text())

    assert previous["properties"]["schema_version"]["const"] == "1.5"
    assert "feeding_max_resource_intake" not in previous["$defs"]["EnergyConfig"]["properties"]


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
    command = (
        "import sys; import evolucio.config; "
        "assert 'jax' not in sys.modules and 'equinox' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
