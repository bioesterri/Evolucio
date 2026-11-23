"""Every founder policy is recoverable with its original inference scores."""

from pathlib import Path

import jax
import numpy as np
import pytest

from evolucio.cli.main import main
from evolucio.core.policy.batch import policy_at
from evolucio.core.policy.schema import POLICY_INPUT_SIZE
from evolucio.persistence.genomes import (
    GenomeHighlightReason,
    genome_content_hash,
    load_highlighted_genome,
    read_genome_index,
    read_genome_leaves,
)
from evolucio.runtime import RunPlan, initialization
from evolucio.runtime.recorded_run import run_recorded_simulation
from tests.zarr_helpers import checkpoint_host_config, checkpoint_scenario


def test_founder_genomes_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, compiled = checkpoint_scenario()
    monkeypatch.setattr(initialization, "initialize_simulation", lambda *_a, **_kw: state)
    run = run_recorded_simulation(
        config=checkpoint_host_config(),
        compiled=compiled,
        output_root=tmp_path,
        plan=RunPlan(1, 1),
    )
    index = read_genome_index(run.run_dir)
    observation = jax.device_put(np.linspace(-1, 1, POLICY_INPUT_SIZE, dtype=np.float32))
    founders = [
        record for record in index.genomes if GenomeHighlightReason.FOUNDER in record.reasons
    ]
    assert len(founders) == 5
    final = run.result.runtime.state
    for record in index.genomes:
        origin = state if record in founders else final
        slot = int(np.flatnonzero(np.asarray(origin.population.agent_id) == record.agent_id)[0])
        original = policy_at(origin.genomes, slot)
        loaded = load_highlighted_genome(run.run_dir, record.genome_id)
        for expected, actual in zip(
            jax.tree.leaves(original), jax.tree.leaves(loaded), strict=True
        ):
            np.testing.assert_array_equal(expected, actual)
        assert genome_content_hash(read_genome_leaves(run.run_dir, record)) == record.genome_hash
        np.testing.assert_array_equal(original(observation), loaded(observation))
    assert main(["genome", "list", str(run.run_dir)]) == 0
    assert main(["genome", "verify", str(run.run_dir)]) == 0
    assert main(["run", "inspect", str(run.run_dir)]) == 0
