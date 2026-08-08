import sys
import types
from unittest.mock import patch, MagicMock, call

import pytest

from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)


POSTGRES_PARAMS = {
    "db_path": "/tmp",
    "db_name": "test_db",
    "db_host": "localhost",
    "db_port": "5432",
    "db_username": "user",
    "db_password": "pass",
    "db_provider": "postgres",
}

TURSO_PARAMS = {
    "db_path": "/tmp",
    "db_name": "test_db",
    "db_host": "",
    "db_port": "",
    "db_username": "",
    "db_password": "",
    "db_provider": "turso",
}


@pytest.fixture(autouse=True)
def _clear_lru_cache():
    """create_relational_engine is wrapped with @lru_cache — clear between tests."""
    create_relational_engine.cache_clear()
    yield
    create_relational_engine.cache_clear()


@pytest.fixture(autouse=True)
def _fake_asyncpg():
    """Inject a fake asyncpg module so the import inside the function succeeds."""
    fake = types.ModuleType("asyncpg")
    with patch.dict(sys.modules, {"asyncpg": fake}):
        yield


class TestCreateRelationalEngineSpecialCharacters:
    """Verify that special characters in credentials are handled correctly by URL.create."""

    @patch("cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter")
    def test_postgres_special_chars_in_username_and_password(self, mock_adapter):
        """Username and password with special characters should round-trip correctly."""
        create_relational_engine(
            **{**POSTGRES_PARAMS, "db_username": "user#name", "db_password": "p@ss:word"}
        )

        url = mock_adapter.call_args[0][0]
        assert url.username == "user#name"
        assert url.password == "p@ss:word"


class TestCreateRelationalEngineConnectArgs:
    """Verify that connect_args are forwarded to the SQLAlchemyAdapter."""

    @patch("cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter")
    def test_postgres_no_connect_args_passes_empty_dict(self, mock_adapter):
        """When no connect_args are provided, an empty dict should be forwarded to the adapter."""
        create_relational_engine(**POSTGRES_PARAMS)

        _, kwargs = mock_adapter.call_args
        assert kwargs.get("connect_args") == {}


class TestCreateRelationalEngineTurso:
    """Verify the DB_PROVIDER=turso branch builds the aiosqlite drop-in URL and returns a TursoAdapter."""

    @pytest.fixture(autouse=True)
    def _fake_libsql(self):
        """Inject a fake libsql_experimental so the driver import inside the branch succeeds."""
        fake = types.ModuleType("libsql_experimental")
        with patch.dict(sys.modules, {"libsql_experimental": fake}):
            yield

    @patch("cognee.infrastructure.databases.relational.sqlalchemy.TursoAdapter.TursoAdapter")
    def test_turso_returns_turso_adapter_with_aiosqlite_url(self, mock_turso_adapter):
        """turso provider builds a local sqlite+aiosqlite connection string and returns a TursoAdapter."""
        engine = create_relational_engine(**TURSO_PARAMS)

        assert engine is mock_turso_adapter.return_value
        connection_string = mock_turso_adapter.call_args[0][0]
        expected = f"sqlite+aiosqlite:///{TURSO_PARAMS['db_path']}/{TURSO_PARAMS['db_name']}"
        assert connection_string == expected

    @patch("cognee.infrastructure.databases.relational.sqlalchemy.TursoAdapter.TursoAdapter")
    def test_turso_remote_forwards_sync_url_and_auth_token(self, mock_turso_adapter):
        """A configured remote URL + auth token are forwarded to the adapter for replica sync."""
        create_relational_engine(
            **{
                **TURSO_PARAMS,
                "db_turso_url": "libsql://db.turso.io",
                "db_turso_auth_token": "tok",
            }
        )

        _, kwargs = mock_turso_adapter.call_args
        assert kwargs.get("sync_url") == "libsql://db.turso.io"
        assert kwargs.get("auth_token") == "tok"

    def test_turso_missing_driver_raises_actionable_import_error(self):
        """When the libSQL driver is missing, raise a clear cognee[turso] install error."""
        # Setting the module to None in sys.modules makes `import libsql_experimental` raise ImportError.
        with patch.dict(sys.modules, {"libsql_experimental": None}):
            with pytest.raises(ImportError, match="Turso/libSQL"):
                create_relational_engine(**TURSO_PARAMS)
