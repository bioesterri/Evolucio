import dataclasses
import re

from evolucio.config import ExperimentConfig, freeze_config, load_config


def test_canonical_hash(config: ExperimentConfig) -> None:
    frozen = freeze_config(config)
    assert re.fullmatch(r"[0-9a-f]{64}", frozen.config_hash)
    assert '"checkpoint_stride":null' in frozen.canonical_json
    assert freeze_config(config) == frozen
    assert frozen.config_hash == "0d796ca9545bc9f0640a69403f21f23fb86dcb9950a9bb11f8940c80b14e54f9"
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        frozen.config_hash = "x"


def test_equivalent_fixtures_and_change(config: ExperimentConfig) -> None:
    other = load_config("tests/fixtures/config/valid_v1.json")
    assert freeze_config(other).config_hash == freeze_config(config).config_hash
    changed = config.model_copy(update={"seed": 43})
    assert freeze_config(changed).config_hash != freeze_config(config).config_hash
