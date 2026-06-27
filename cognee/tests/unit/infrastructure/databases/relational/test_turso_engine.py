import sys
import types
from unittest.mock import patch, MagicMock

import pytest

from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)


TURSO_REMOTE_PARAMS = {
    "db_path": "/tmp",
    "db_name": "test_db",
    "db_host": "localhost",
    "db_port": "5432",
    "db_username": "user",
    "db_password": "pass",
    "db_provider": "turso",
    "turso_url": "libsql://mydb-myorg.turso.io",
    "turso_auth_token": "my-secret-token",
}

TURSO_LOCAL_PARAMS = {
    "db_path": "/tmp/cognee",
    "db_name": "cognee_db",
    "db_host": None,
    "db_port": None,
    "db_username": None,
    "db_password": None,
    "db_provider": "turso",
    "turso_url": None,
    "turso_auth_token": None,
}


@pytest.fixture(autouse=True)
def _clear_lru_cache():
    """create_relational_engine is wrapped with @lru_cache — clear between tests."""
    create_relational_engine.cache_clear()
    yield
    create_relational_engine.cache_clear()


@pytest.fixture(autouse=True)
def _fake_sqlalchemy_libsql():
    """Inject a fake sqlalchemy_libsql module so the import inside the function succeeds."""
    fake = types.ModuleType("sqlalchemy_libsql")
    with patch.dict(sys.modules, {"sqlalchemy_libsql": fake}):
        yield


class TestTursoRemoteConnectionString:
    """Verify that remote Turso builds the correct libsql:// connection string."""

    @patch(
        "cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter"
    )
    def test_turso_remote_builds_correct_url(self, mock_adapter):
        """Remote Turso should produce a libsql:// URL with authToken query param."""
        create_relational_engine(**TURSO_REMOTE_PARAMS)

        connection_string = mock_adapter.call_args[0][0]
        assert "libsql://mydb-myorg.turso.io" in connection_string
        assert "authToken=my-secret-token" in connection_string
        assert "secure=true" in connection_string

    @patch(
        "cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter"
    )
    def test_turso_remote_returns_sqlalchemy_adapter(self, mock_adapter):
        """The factory should return a SQLAlchemyAdapter instance for Turso."""
        result = create_relational_engine(**TURSO_REMOTE_PARAMS)
        assert result == mock_adapter.return_value


class TestTursoLocalConnectionString:
    """Verify that local/embedded Turso builds a sqlite+libsql:// connection string."""

    @patch(
        "cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter"
    )
    def test_turso_local_builds_sqlite_libsql_url(self, mock_adapter):
        """Local Turso (no turso_url) should produce a sqlite+libsql:/// file path."""
        create_relational_engine(**TURSO_LOCAL_PARAMS)

        connection_string = mock_adapter.call_args[0][0]
        assert connection_string == "sqlite+libsql:////tmp/cognee/cognee_db"


class TestTursoMissingDependency:
    """Verify that missing sqlalchemy-libsql raises an actionable ImportError."""

    def test_turso_missing_driver_raises_import_error(self):
        """When sqlalchemy_libsql is not installed, an ImportError with install instructions
        should be raised."""
        with patch.dict(sys.modules, {"sqlalchemy_libsql": None}):
            with pytest.raises(ImportError, match=r"pip install cognee.*turso"):
                create_relational_engine(**TURSO_REMOTE_PARAMS)


class TestTursoConnectArgs:
    """Verify that connect_args are forwarded to the SQLAlchemyAdapter."""

    @patch(
        "cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter"
    )
    def test_turso_no_connect_args_passes_empty_dict(self, mock_adapter):
        """When no connect_args are provided, an empty dict should be forwarded."""
        create_relational_engine(**TURSO_REMOTE_PARAMS)

        _, kwargs = mock_adapter.call_args
        assert kwargs.get("connect_args") == {}
