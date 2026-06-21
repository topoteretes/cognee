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


class TestCreateRelationalEngineSqliteDirectory:
    """Verify that local SQLite database directories are created before connecting."""

    @patch("cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter")
    def test_sqlite_creates_missing_database_directory(self, mock_adapter, tmp_path):
        db_path = tmp_path / "missing" / "databases"

        create_relational_engine(
            db_path=str(db_path),
            db_name="cognee_db",
            db_host=None,
            db_port=None,
            db_username=None,
            db_password=None,
            db_provider="sqlite",
        )

        assert db_path.is_dir()
        mock_adapter.assert_called_once()
        assert mock_adapter.call_args.args[0] == f"sqlite+aiosqlite:///{db_path}/cognee_db"


class TestCreateRelationalEngineConnectArgs:
    """Verify that connect_args are forwarded to the SQLAlchemyAdapter."""

    @patch("cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter")
    def test_postgres_no_connect_args_passes_empty_dict(self, mock_adapter):
        """When no connect_args are provided, an empty dict should be forwarded to the adapter."""
        create_relational_engine(**POSTGRES_PARAMS)

        _, kwargs = mock_adapter.call_args
        assert kwargs.get("connect_args") == {}
