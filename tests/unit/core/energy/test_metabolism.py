import equinox as eqx
import jax
import jax.numpy as jnp

from evolucio.core.energy import apply_basal_metabolism_and_age
from evolucio.core.state import PopulationState
from evolucio.core.types import Array

from .conftest import population


def test_basal_age_inactive_negative_energy_and_no_death() -> None:
    state = population([1.0, 7.0], [True, False], [4, 8])
    result = apply_basal_metabolism_and_age(state, basal_metabolic_cost=jnp.float32(2))
    assert result.population.energy.tolist() == [-1, 7]
    assert result.population.age.tolist() == [5, 8]
    assert result.population.alive.tolist() == [True, False]
    assert result.basal_cost_applied.tolist() == [2, 0]
    assert result.age_incremented.tolist() == [True, False]


def test_invalid_live_values_are_preserved_and_counted() -> None:
    state = population([jnp.nan, jnp.inf, 3], [True, True, True], [-1, 2**31 - 1, 3])
    result = apply_basal_metabolism_and_age(state, basal_metabolic_cost=jnp.float32(1))
    assert jnp.isnan(result.population.energy[0]) and jnp.isinf(result.population.energy[1])
    assert result.population.age.tolist() == [-1, 2**31 - 1, 4]
    assert result.invalid_active_energy_count.item() == 2
    assert result.invalid_active_age_count.item() == 2


def test_metabolism_eager_jit_scan_and_fixed_shapes() -> None:
    state = population([4, 5, 6], [True, False, True])
    cost = jnp.float32(0.5)
    eager = apply_basal_metabolism_and_age(state, basal_metabolic_cost=cost)
    compiled = eqx.filter_jit(apply_basal_metabolism_and_age)(state, basal_metabolic_cost=cost)
    assert jnp.array_equal(eager.population.energy, compiled.population.energy)

    def body(carry: PopulationState, _: None) -> tuple[PopulationState, Array]:
        result = apply_basal_metabolism_and_age(carry, basal_metabolic_cost=cost)
        return result.population, result.basal_cost_applied

    scanned, costs = jax.lax.scan(body, state, xs=None, length=2)
    assert scanned.energy.shape == (3,)
    assert costs.shape == (2, 3)
