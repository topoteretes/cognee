from unittest.mock import patch
from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)


class TestSqlAlchemyAdapter:
    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    @patch("os.getenv")
    def test_sqlite_default_timeout(self, mock_getenv, mock_create_engine):
        """Test that SQLite connection uses default timeout when no env var is set."""
        mock_getenv.return_value = None
        SQLAlchemyAdapter("sqlite:///test.db")
        mock_create_engine.assert_called_once()
        _, kwargs = mock_create_engine.call_args
        assert "connect_args" in kwargs
        assert kwargs["connect_args"] == {"timeout": 30}

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    @patch("os.getenv")
    def test_sqlite_with_env_var_timeout(self, mock_getenv, mock_create_engine):
        """Test that SQLite connection uses timeout from env var."""
        mock_getenv.return_value = '{"timeout": 60}'
        SQLAlchemyAdapter("sqlite:///test.db")
        mock_create_engine.assert_called_once()
        _, kwargs = mock_create_engine.call_args
        assert "connect_args" in kwargs
        assert kwargs["connect_args"] == {"timeout": 60}

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    @patch("os.getenv")
    def test_sqlite_with_other_env_var_args(self, mock_getenv, mock_create_engine):
        """Test that SQLite connection merges default timeout with other args from env var."""
        mock_getenv.return_value = '{"foo": "bar"}'
        SQLAlchemyAdapter("sqlite:///test.db")
        mock_create_engine.assert_called_once()
        _, kwargs = mock_create_engine.call_args
        assert "connect_args" in kwargs
        assert kwargs["connect_args"] == {"timeout": 30, "foo": "bar"}

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    @patch("cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.logger")
    @patch("os.getenv")
    def test_sqlite_with_invalid_json_env_var(self, mock_getenv, mock_logger, mock_create_engine):
        """Test that SQLite connection uses default timeout when env var has invalid JSON."""
        mock_getenv.return_value = '{"timeout": 60'  # Invalid JSON
        SQLAlchemyAdapter("sqlite:///test.db")

        mock_logger.warning.assert_called_with(
            "Failed to parse DATABASE_CONNECT_ARGS as JSON, ignoring"
        )

        mock_create_engine.assert_called_once()
        _, kwargs = mock_create_engine.call_args
        assert "connect_args" in kwargs
        assert kwargs["connect_args"] == {"timeout": 30}

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    @patch("os.getenv")
    def test_postgresql_with_env_var(self, mock_getenv, mock_create_engine):
        """Test that PostgreSQL connection uses connect_args from env var."""
        mock_getenv.return_value = '{"sslmode": "require"}'
        SQLAlchemyAdapter("postgresql://user:pass@host/db")
        mock_create_engine.assert_called_once()
        _, kwargs = mock_create_engine.call_args
        assert "connect_args" in kwargs
        assert kwargs["connect_args"] == {"sslmode": "require"}

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    @patch("os.getenv")
    def test_postgresql_without_env_var(self, mock_getenv, mock_create_engine):
        """Test that PostgreSQL connection has empty connect_args when no env var is set."""
        mock_getenv.return_value = None
        SQLAlchemyAdapter("postgresql://user:pass@host/db")
        mock_create_engine.assert_called_once()
        _, kwargs = mock_create_engine.call_args
        assert "connect_args" in kwargs
        assert kwargs["connect_args"] == {}
