import sys
import types
from unittest.mock import patch, MagicMock

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


class TestCreateRelationalEngineSSL:
    """Verify that the SSL query parameter is handled correctly in the connection URL."""

    @patch(
        "cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter"
    )
    def test_postgres_without_ssl(self, mock_adapter):
        """When db_ssl_mode is None, the URL must NOT contain an ssl query param."""
        create_relational_engine(**POSTGRES_PARAMS, db_ssl_mode=None)

        url = mock_adapter.call_args[0][0]
        assert "ssl" not in url.query

    @patch(
        "cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter"
    )
    def test_postgres_with_ssl_require(self, mock_adapter):
        """When db_ssl_mode='require', the URL must contain ssl=require."""
        create_relational_engine(**POSTGRES_PARAMS, db_ssl_mode="require")

        url = mock_adapter.call_args[0][0]
        assert url.query["ssl"] == "require"

    @patch(
        "cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter"
    )
    def test_postgres_with_ssl_verify_full(self, mock_adapter):
        """When db_ssl_mode='verify-full', the URL must contain ssl=verify-full."""
        create_relational_engine(**POSTGRES_PARAMS, db_ssl_mode="verify-full")

        url = mock_adapter.call_args[0][0]
        assert url.query["ssl"] == "verify-full"

    @patch(
        "cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter"
    )
    def test_postgres_default_has_no_ssl(self, mock_adapter):
        """When db_ssl_mode is omitted entirely, the URL must NOT contain ssl."""
        create_relational_engine(**POSTGRES_PARAMS)

        url = mock_adapter.call_args[0][0]
        assert "ssl" not in url.query
