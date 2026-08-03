import ast
from pathlib import Path


def test_observation_modules_avoid_rng_identity_and_forbidden_layers() -> None:
    root = Path("src/evolucio/core/observations")
    forbidden_text = (
        "jax.random",
        "numpy.random",
        "np.random",
        "random.",
        "agent_id",
        "parent_id",
        "lineage_id",
        "genome_id",
    )
    forbidden_imports = (
        "policy",
        "actions",
        "evolution",
        "metrics",
        "engine",
        "runtime",
        "persistence",
        "analysis",
        "visualization",
    )
    for path in root.glob("*.py"):
        source = path.read_text()
        assert not any(term in source for term in forbidden_text)
        tree = ast.parse(source)
        modules = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        assert not any(
            any(part in module.split(".") for part in forbidden_imports) for module in modules
        )
