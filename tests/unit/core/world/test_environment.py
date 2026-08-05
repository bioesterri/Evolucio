import equinox as eqx
import jax.numpy as jnp
import pytest

from evolucio.config import ExperimentConfig, compile_config, load_config
from evolucio.core import INDEX_DTYPE, REAL_DTYPE, STEP_DTYPE
from evolucio.core.world import (
    NO_ACTIVE_PHASE,
    resolve_environment_control,
    update_environment_layer,
)


def calendar_config() -> object:
    config = load_config("tests/fixtures/config/valid_v1.yaml")
    raw = config.model_dump(mode="python")
    raw["world"]["environment_initial_value"] = 0.1  # type: ignore[index]
    raw["world"]["environment_schedule"] = [  # type: ignore[index]
        {"start_step": 10, "end_step": 15, "regeneration_multiplier": 0.25, "stress_level": 0.8},
        {"start_step": 20, "end_step": 30, "regeneration_multiplier": 0.6, "stress_level": 0.4},
    ]
    return compile_config(ExperimentConfig.model_validate(raw)).core.world


@pytest.mark.parametrize(
    ("step", "index", "multiplier", "value"),
    [
        (0, -1, 1, 0.1),
        (9, -1, 1, 0.1),
        (10, 0, 0.25, 0.8),
        (14, 0, 0.25, 0.8),
        (15, -1, 1, 0.1),
        (20, 1, 0.6, 0.4),
        (29, 1, 0.6, 0.4),
        (30, -1, 1, 0.1),
    ],
)
def test_resolve_half_open_calendar(step: int, index: int, multiplier: float, value: float) -> None:
    config = calendar_config()
    step_array = jnp.asarray(step, dtype=STEP_DTYPE)
    eager = resolve_environment_control(step_array, config)  # type: ignore[arg-type]
    compiled = eqx.filter_jit(resolve_environment_control)(step_array, config)
    assert int(eager.active_phase_index) == index
    assert float(eager.regeneration_multiplier) == pytest.approx(multiplier)
    assert float(eager.environment_value) == pytest.approx(value)
    assert eager.active_phase_index.shape == () and eager.active_phase_index.dtype == jnp.dtype(
        INDEX_DTYPE
    )
    assert eager.environment_value.dtype == jnp.dtype(REAL_DTYPE)
    assert jnp.array_equal(eager.active_phase_index, compiled.active_phase_index)


def test_empty_calendar_and_environment_fill() -> None:
    config = compile_config(load_config("tests/fixtures/config/valid_v1.yaml")).core.world
    control = resolve_environment_control(jnp.asarray(0, STEP_DTYPE), config)
    layer = update_environment_layer(jnp.ones((2, 3), REAL_DTYPE), jnp.asarray(0.3, REAL_DTYPE))
    assert int(control.active_phase_index) == NO_ACTIVE_PHASE
    assert layer.shape == (2, 3) and layer.dtype == jnp.dtype(REAL_DTYPE)
    assert jnp.allclose(layer, 0.3)
