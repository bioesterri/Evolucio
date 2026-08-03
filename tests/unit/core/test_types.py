import ast
from pathlib import Path

import jax

import evolucio.core
from evolucio.core import AgentId, Array, GenomeId, LineageId, Shape, StepIndex


def test_public_type_aliases() -> None:
    assert Array.__value__ is jax.Array
    assert Shape.__value__ == tuple[int, ...]
    assert all(alias.__value__ is int for alias in (AgentId, GenomeId, LineageId, StepIndex))


def test_core_has_no_reverse_dependency_on_config() -> None:
    core_root = Path(evolucio.core.__file__).parent
    imported_modules = {
        node.module
        for path in core_root.glob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(module == "evolucio.config" for module in imported_modules)
