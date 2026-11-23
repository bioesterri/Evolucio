# ruff: noqa: ANN001, ANN201
import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url


@pytest.fixture(scope="session")
def database_url():
    url = os.environ.get("EVOLUCIO_TEST_DATABASE_URL")
    if not url:
        pytest.skip("EVOLUCIO_TEST_DATABASE_URL is not configured")
    parsed = make_url(url)
    if parsed.drivername != "postgresql+psycopg" or not (
        parsed.database and parsed.database.lower().endswith("_test")
    ):
        pytest.fail("integration tests require an explicit PostgreSQL test database URL")
    return url


@pytest.fixture
def migrated_engine(database_url, monkeypatch):
    monkeypatch.setenv("EVOLUCIO_DATABASE_URL", database_url)
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    yield engine
    engine.dispose()
