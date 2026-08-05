"""Test postgres-graph connection resolution via the shared Postgres resolver.

Verifies that when the postgres graph provider configures no credentials of its
own, the engine factory falls back to the unified ``DB_*`` relational block —
silently (no fallback warning), now that the shared block is the documented
default. Also confirms the internal dataset handler resolves to ``postgres_graph``
while the user-facing provider stays ``postgres``.
"""

import os
import sys
import types
from unittest.mock import patch

import pytest

from cognee.infrastructure.databases.graph.get_graph_engine import (
    create_graph_engine,
    _create_graph_engine,
)
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.relational import get_relational_config


# Only the shared DB_* block plus the postgres provider are configured — no
# GRAPH_DATABASE_* credential parts, so resolution must fall back to DB_*.
# GRAPH_DATASET_DATABASE_HANDLER is pinned to ladybug here because the repo .env
# sets it to "kuzu", which would otherwise block the postgres_graph mapping.
DB_ENV = {
    "USE_UNIFIED_PROVIDER": "",
    "ENABLE_BACKEND_ACCESS_CONTROL": "false",
    "GRAPH_DATABASE_PROVIDER": "postgres",
    "GRAPH_DATASET_DATABASE_HANDLER": "ladybug",
    "DB_HOST": "shared-host",
    "DB_PORT": "5432",
    "DB_USERNAME": "shared_user",
    "DB_PASSWORD": "shared_pass",
    "DB_NAME": "shared_db",
}


@pytest.fixture(autouse=True)
def _clear_caches():
    """The engine factory and both configs are cached — clear between tests."""
    _create_graph_engine.cache_clear()
    get_graph_config.cache_clear()
    get_relational_config.cache_clear()
    yield
    _create_graph_engine.cache_clear()
    get_graph_config.cache_clear()
    get_relational_config.cache_clear()


@pytest.fixture
def _fake_postgres_adapter():
    """Inject a fake PostgresAdapter module so the import inside the engine
    factory succeeds without the postgres extra installed, and capture its args.
    """
    captured = {}

    class FakePostgresAdapter:
        def __init__(self, connection_string):
            captured["connection_string"] = connection_string

    fake_module = types.ModuleType("cognee.infrastructure.databases.graph.postgres.adapter")
    fake_module.PostgresAdapter = FakePostgresAdapter

    with patch.dict(
        sys.modules,
        {"cognee.infrastructure.databases.graph.postgres.adapter": fake_module},
    ):
        yield captured


def test_postgres_graph_falls_back_to_shared_db_block_without_warning(_fake_postgres_adapter):
    """Only DB_* set -> adapter gets a URL carrying the DB_* credentials, no warning."""
    with patch.dict(os.environ, DB_ENV):
        get_graph_config.cache_clear()
        get_relational_config.cache_clear()

        config = get_graph_config()
        # Internal dataset handler maps to postgres_graph; user-facing provider stays postgres.
        assert config.graph_database_provider == "postgres"
        assert config.graph_dataset_database_handler == "postgres_graph"

        params = config.to_hashable_dict()

        with patch("cognee.infrastructure.databases.graph.get_graph_engine.logger") as mock_logger:
            create_graph_engine(**params)

    url = _fake_postgres_adapter["connection_string"]
    assert url.host == "shared-host"
    assert url.port == 5432
    assert url.username == "shared_user"
    assert url.password == "shared_pass"
    assert url.database == "shared_db"

    # The fallback to the shared DB_* block is now the documented default and
    # must not emit a warning.
    mock_logger.warning.assert_not_called()
