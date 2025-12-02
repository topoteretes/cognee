import os
from unittest.mock import patch
from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)


class TestSqlAlchemyAdapter:
    """Test suite for SqlAlchemyAdapter environment variable handling and connection arguments."""

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    def test_sqlite_default_timeout(self, mock_create_engine):
        """Test that SQLite connection uses default timeout when no env var is set."""
        with patch.dict(os.environ, {}, clear=True):
            SQLAlchemyAdapter("sqlite:///test.db")
            mock_create_engine.assert_called_once()
            _, kwargs = mock_create_engine.call_args
            assert "connect_args" in kwargs
            assert kwargs["connect_args"] == {"timeout": 30}

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    def test_sqlite_with_env_var_timeout(self, mock_create_engine):
        """Test that SQLite connection uses timeout from env var."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '{"timeout": 60}'}):
            SQLAlchemyAdapter("sqlite:///test.db")
            mock_create_engine.assert_called_once()
            _, kwargs = mock_create_engine.call_args
            assert "connect_args" in kwargs
            assert kwargs["connect_args"] == {"timeout": 60}

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    def test_sqlite_with_other_env_var_args(self, mock_create_engine):
        """Test that SQLite connection merges default timeout with other args from env var."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '{"foo": "bar"}'}):
            SQLAlchemyAdapter("sqlite:///test.db")
            mock_create_engine.assert_called_once()
            _, kwargs = mock_create_engine.call_args
            assert "connect_args" in kwargs
            assert kwargs["connect_args"] == {"timeout": 30, "foo": "bar"}

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    @patch("cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.logger")
    def test_sqlite_with_invalid_json_env_var(self, mock_logger, mock_create_engine):
        """Test that SQLite connection uses default timeout when env var has invalid JSON."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '{"timeout": 60'}):  # Invalid JSON
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
    @patch("cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.logger")
    def test_sqlite_with_non_dict_json_env_var(self, mock_logger, mock_create_engine):
        """Test that SQLite connection uses default timeout when env var is valid JSON but not a dict."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '["list", "instead", "of", "dict"]'}):
            SQLAlchemyAdapter("sqlite:///test.db")

            mock_logger.warning.assert_called_with(
                "DATABASE_CONNECT_ARGS is not a valid JSON dictionary, ignoring"
            )

            mock_create_engine.assert_called_once()
            _, kwargs = mock_create_engine.call_args
            assert "connect_args" in kwargs
            assert kwargs["connect_args"] == {"timeout": 30}

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    def test_postgresql_with_env_var(self, mock_create_engine):
        """Test that PostgreSQL connection uses connect_args from env var."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '{"sslmode": "require"}'}):
            SQLAlchemyAdapter("postgresql://user:pass@host/db")
            mock_create_engine.assert_called_once()
            _, kwargs = mock_create_engine.call_args
            assert "connect_args" in kwargs
            assert kwargs["connect_args"] == {"sslmode": "require"}

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    def test_postgresql_without_env_var(self, mock_create_engine):
        """Test that PostgreSQL connection has empty connect_args when no env var is set."""
        with patch.dict(os.environ, {}, clear=True):
            SQLAlchemyAdapter("postgresql://user:pass@host/db")
            mock_create_engine.assert_called_once()
            _, kwargs = mock_create_engine.call_args
            assert "connect_args" in kwargs
            assert kwargs["connect_args"] == {}

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    def test_connect_args_precedence(self, mock_create_engine):
        """Test that explicit connect_args take precedence over env var args."""
        with patch.dict(os.environ, {"DATABASE_CONNECT_ARGS": '{"timeout": 60}'}):
            # Pass explicit connect_args that should override env var
            SQLAlchemyAdapter("sqlite:///test.db", connect_args={"timeout": 120})

            mock_create_engine.assert_called_once()
            _, kwargs = mock_create_engine.call_args
            assert "connect_args" in kwargs
            # timeout should be 120 (explicit), not 60 (env var) or 30 (default)
            assert kwargs["connect_args"] == {"timeout": 120}
