from unittest.mock import patch, MagicMock

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)


class TestSQLAlchemyAdapterConnectArgs:
    """Verify that connect_args is only passed to create_async_engine when non-empty."""

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    def test_empty_connect_args_not_passed_to_engine(self, mock_create_engine):
        """When connect_args is None/empty, create_async_engine should NOT receive connect_args."""
        mock_create_engine.return_value = MagicMock()

        SQLAlchemyAdapter("postgresql+asyncpg://user:pass@localhost:5432/db")

        _, kwargs = mock_create_engine.call_args
        assert "connect_args" not in kwargs

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    def test_non_empty_connect_args_passed_to_engine(self, mock_create_engine):
        """When connect_args has values, create_async_engine should receive them."""
        mock_create_engine.return_value = MagicMock()

        custom_args = {"sslmode": "require"}
        SQLAlchemyAdapter(
            "postgresql+asyncpg://user:pass@localhost:5432/db",
            connect_args=custom_args,
        )

        _, kwargs = mock_create_engine.call_args
        assert kwargs["connect_args"] == {"sslmode": "require"}

    @patch(
        "cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter.create_async_engine"
    )
    def test_sqlite_always_passes_connect_args_with_timeout(self, mock_create_engine):
        """SQLite should always pass connect_args with at least the timeout default."""
        mock_create_engine.return_value = MagicMock()

        SQLAlchemyAdapter("sqlite+aiosqlite:///tmp/test.db")

        _, kwargs = mock_create_engine.call_args
        assert "timeout" in kwargs["connect_args"]
        assert kwargs["connect_args"]["timeout"] == 30
