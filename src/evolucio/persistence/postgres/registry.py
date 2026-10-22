"""Engine lifecycle and PostgreSQL/Alembic health operations."""

import os
from dataclasses import dataclass
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .errors import DatabaseSchemaMismatchError, DatabaseUnavailableError

DATABASE_URL_ENV = "EVOLUCIO_DATABASE_URL"
ALEMBIC_INI = Path(__file__).parents[4] / "alembic.ini"
POSTGRES_URL_PREFIX = "postgresql+psycopg://"


@dataclass(frozen=True, slots=True)
class DatabaseStatus:
    current_revision: str | None
    head_revision: str
    up_to_date: bool


def database_url_from_environment(variable: str = DATABASE_URL_ENV) -> str:
    value = os.environ.get(variable)
    if not value:
        raise DatabaseUnavailableError(f"{variable} is not configured")
    if not value.startswith(POSTGRES_URL_PREFIX):
        raise DatabaseUnavailableError("the registry requires a PostgreSQL SQLAlchemy URL")
    return value


def create_registry_engine(database_url: str | None = None) -> Engine:
    url = database_url or database_url_from_environment()
    if not url.startswith(POSTGRES_URL_PREFIX):
        raise DatabaseUnavailableError("the registry requires PostgreSQL; SQLite is unsupported")
    return create_engine(url, pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def check_database_connection(engine: Engine) -> None:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError("PostgreSQL connection check failed") from exc


def _alembic_config(database_url: str | None = None) -> Config:
    config = Config(str(ALEMBIC_INI))
    if database_url is not None:
        config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def database_status(engine: Engine) -> DatabaseStatus:
    config = _alembic_config(str(engine.url))
    head = ScriptDirectory.from_config(config).get_current_head()
    if head is None:
        raise DatabaseSchemaMismatchError("Alembic does not define a head revision")
    with engine.connect() as connection:
        current = MigrationContext.configure(connection).get_current_revision()
    return DatabaseStatus(current, head, current == head)


def upgrade_database(database_url: str | None = None) -> None:
    url = database_url or database_url_from_environment()
    try:
        command.upgrade(_alembic_config(url), "head")
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError("database migration failed") from exc
