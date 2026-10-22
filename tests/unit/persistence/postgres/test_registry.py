# ruff: noqa: ANN001, ANN201
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from evolucio.persistence.postgres.errors import DatabaseUnavailableError
from evolucio.persistence.postgres.registry import (
    check_database_connection,
    create_registry_engine,
    database_url_from_environment,
)
from evolucio.persistence.postgres.types import RunSummaryRecord


def test_database_url_is_explicit_and_postgres(monkeypatch):
    monkeypatch.delenv("EVOLUCIO_DATABASE_URL", raising=False)
    with pytest.raises(DatabaseUnavailableError):
        database_url_from_environment()
    with pytest.raises(DatabaseUnavailableError):
        create_registry_engine("sqlite+pysqlite:///:memory:")


def test_connection_error_is_contextualized():
    failing = Mock()
    failing.connect.side_effect = SQLAlchemyError("down")
    with pytest.raises(DatabaseUnavailableError):
        check_database_connection(failing)


def test_summary_derives_extinction_only_from_final_count():
    alive = RunSummaryRecord(2, 3, 1, 1, 4.0, 5.0)
    extinct = RunSummaryRecord(0, 3, 3, 0, 0.0, 5.0)
    assert not alive.population_extinct
    assert extinct.population_extinct
