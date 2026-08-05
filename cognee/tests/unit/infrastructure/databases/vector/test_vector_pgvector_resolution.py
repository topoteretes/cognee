"""Test pgvector connection resolution via the shared Postgres resolver.

Verifies that when a pgvector store configures no credentials of its own, the
engine factory falls back to the unified ``DB_*`` relational block — and does so
silently (no fallback warning), now that the resolver makes the shared block the
documented default rather than a last-resort path.
"""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from cognee.infrastructure.databases.vector.create_vector_engine import (
    create_vector_engine,
    _create_vector_engine,
)
from cognee.infrastructure.databases.vector.config import get_vectordb_config
from cognee.infrastructure.databases.relational import get_relational_config


# Only the shared DB_* block plus the pgvector provider are configured — no
# VECTOR_DB_* credential parts, so resolution must fall back to DB_*.
DB_ENV = {
    "USE_UNIFIED_PROVIDER": "",
    "ENABLE_BACKEND_ACCESS_CONTROL": "false",
    "VECTOR_DB_PROVIDER": "pgvector",
    "DB_HOST": "shared-host",
    "DB_PORT": "5432",
    "DB_USERNAME": "shared_user",
    "DB_PASSWORD": "shared_pass",
    "DB_NAME": "shared_db",
}


@pytest.fixture(autouse=True)
def _clear_caches():
    """The engine factory and both configs are cached — clear between tests."""
    _create_vector_engine.cache_clear()
    get_vectordb_config.cache_clear()
    get_relational_config.cache_clear()
    yield
    _create_vector_engine.cache_clear()
    get_vectordb_config.cache_clear()
    get_relational_config.cache_clear()


@pytest.fixture
def _fake_pgvector_adapter():
    """Inject a fake PGVectorAdapter module so the import inside the engine
    factory succeeds without the postgres extra installed, and capture its args.
    """
    captured = {}

    class FakePGVectorAdapter:
        def __init__(self, connection_string, api_key, embedding_engine):
            captured["connection_string"] = connection_string
            captured["api_key"] = api_key

    fake_module = types.ModuleType(
        "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter"
    )
    fake_module.PGVectorAdapter = FakePGVectorAdapter

    with patch.dict(
        sys.modules,
        {"cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter": fake_module},
    ):
        yield captured


def test_pgvector_falls_back_to_shared_db_block_without_warning(_fake_pgvector_adapter):
    """Only DB_* set -> adapter gets a URL carrying the DB_* credentials, no warning."""
    with patch.dict(os.environ, DB_ENV):
        get_vectordb_config.cache_clear()
        get_relational_config.cache_clear()
        params = get_vectordb_config().to_dict()

        with (
            patch(
                "cognee.infrastructure.databases.vector.create_vector_engine.get_embedding_engine",
                return_value=MagicMock(),
            ),
            patch(
                "cognee.infrastructure.databases.vector.create_vector_engine.logger"
            ) as mock_logger,
        ):
            create_vector_engine(**params)

    url = _fake_pgvector_adapter["connection_string"]
    assert url.host == "shared-host"
    assert url.port == 5432
    assert url.username == "shared_user"
    assert url.password == "shared_pass"
    assert url.database == "shared_db"

    # The fallback to the shared DB_* block is now the documented default and
    # must not emit a warning.
    mock_logger.warning.assert_not_called()
