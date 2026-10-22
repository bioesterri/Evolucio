# ruff: noqa: ANN201
"""Schema declaration tests that need no database service."""

from sqlalchemy.dialects.postgresql import JSONB

from evolucio.persistence.postgres.models import Base, ConfigSnapshotModel


def test_relational_tables_and_jsonb_are_declared():
    assert set(Base.metadata.tables) == {
        "experiment",
        "run",
        "config_snapshot",
        "run_summary",
        "artifact",
        "dataset_index",
        "state_capture",
    }
    assert isinstance(ConfigSnapshotModel.__table__.c.config_json.type, JSONB)


def test_foreign_key_delete_policies_and_indexes():
    run = Base.metadata.tables["run"]
    assert next(iter(run.c.experiment_id.foreign_keys)).ondelete == "RESTRICT"
    for name in ("config_snapshot", "run_summary", "artifact", "dataset_index", "state_capture"):
        table = Base.metadata.tables[name]
        assert next(iter(table.c.run_id.foreign_keys)).ondelete == "CASCADE"
    names = {index.name for index in run.indexes}
    assert {"ix_run_experiment_config", "ix_run_config_seed"} <= names
