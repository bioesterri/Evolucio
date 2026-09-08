# ruff: noqa: ANN001, ANN201
import jax.numpy as jnp

from evolucio.core.codes import DeathCauseCode
from evolucio.core.dtypes import CODE_DTYPE, STEP_DTYPE
from evolucio.core.viability import DeathRecordBatch, build_death_records


def test_records_snapshot_before_cleanup_and_canonicalise_non_deaths(viability_state):
    population, _, _ = viability_state
    records = build_death_records(
        population=population,
        dies=jnp.asarray([False, True, False]),
        cause=jnp.asarray([0, DeathCauseCode.ENERGY_DEPLETION, 0], dtype=CODE_DTYPE),
        step=jnp.asarray(8, dtype=STEP_DTYPE),
    )
    assert isinstance(records, DeathRecordBatch)
    assert records.died.tolist() == [False, True, False]
    assert records.agent_id.tolist() == [-1, 11, -1]
    assert records.parent_id.tolist() == [-1, 10, -1]
    assert records.position.tolist() == [[-1, -1], [1, 1], [-1, -1]]
    assert records.energy.tolist() == [0.0, 5.0, 0.0]
    assert records.age.tolist() == [0, 3, 0]
    assert records.death_step.tolist() == [0, 8, 0]
    assert records.cause.tolist() == [0, int(DeathCauseCode.ENERGY_DEPLETION), 0]
