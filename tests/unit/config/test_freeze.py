import dataclasses
import re

from evolucio.config import ExperimentConfig, freeze_config, load_config


def test_canonical_hash(config: ExperimentConfig) -> None:
    frozen = freeze_config(config)
    assert re.fullmatch(r"[0-9a-f]{64}", frozen.config_hash)
    assert '"checkpoint_stride":null' in frozen.canonical_json
    assert freeze_config(config) == frozen
    assert frozen.config_hash == "e0e590b23959b224b7438bc99f2e83f7affd7697de970e44e80613bd0e6fa896"
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        frozen.config_hash = "x"


def test_equivalent_fixtures_and_change(config: ExperimentConfig) -> None:
    other = load_config("tests/fixtures/config/valid_v1.json")
    assert freeze_config(other).config_hash == freeze_config(config).config_hash
    changed = config.model_copy(update={"seed": 43})
    assert freeze_config(changed).config_hash != freeze_config(config).config_hash
