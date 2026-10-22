# ruff: noqa: ANN001, ANN201
import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine


@pytest.fixture(scope="session")
def database_url():
    url = os.environ.get("EVOLUCIO_TEST_DATABASE_URL")
    if not url:
        pytest.skip("EVOLUCIO_TEST_DATABASE_URL is not configured")
    if "test" not in url.lower() or not url.startswith("postgresql+"):
        pytest.fail("integration tests require an explicit PostgreSQL test database URL")
    return url


@pytest.fixture
def migrated_engine(database_url):
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    yield engine
    engine.dispose()
