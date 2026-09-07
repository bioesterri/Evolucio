import equinox as eqx
import jax.numpy as jnp

from evolucio.core.energy import build_energy_balance_report


def test_balance_identity_mask_totals_and_jit() -> None:
    before = jnp.asarray([10, 20, 30], jnp.float32)
    basal = jnp.asarray([1, 1, 0], jnp.float32)
    gain = jnp.asarray([4, 0, 100], jnp.float32)
    movement = jnp.asarray([2, 0, 0], jnp.float32)
    feeding = jnp.asarray([3, 0, 0], jnp.float32)
    after = before - basal + gain - movement - feeding
    kwargs = dict(
        energy_before=before,
        energy_after=after,
        basal_cost_applied=basal,
        feeding_energy_gained=gain,
        movement_cost_applied=movement,
        feeding_cost_applied=feeding,
        tracked_mask=jnp.asarray([True, True, False]),
    )
    report = build_energy_balance_report(**kwargs)
    compiled = eqx.filter_jit(build_energy_balance_report)(**kwargs)
    assert report.max_abs_residual.item() == 0
    assert report.residual.tolist() == [0, 0, 0]
    assert report.total_feeding_gain.item() == 4
    assert jnp.array_equal(report.expected_energy_after, compiled.expected_energy_after)
