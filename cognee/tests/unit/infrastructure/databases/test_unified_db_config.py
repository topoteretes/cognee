"""Headline acceptance test: one DB_* Postgres block configures all three stores.

A single shared ``DB_*`` credential block (no per-store credential repetition)
must drive the relational engine, the pgvector vector engine, and the
postgres-graph engine — all resolving to the same Postgres credentials, with no
fallback/typo warnings.
"""

import logging
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from cognee.infrastructure.databases.relational.config import get_relational_config
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.infrastructure.databases.vector.config import get_vectordb_config
from cognee.infrastructure.databases.vector.create_vector_engine import (
    create_vector_engine,
    _create_vector_engine,
)
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import (
    create_graph_engine,
    _create_graph_engine,
)


# A single shared Postgres block — NO VECTOR_DB_*/GRAPH_DATABASE_* credential
# repetition. All three providers point at "postgres" (vector aliases pgvector).
DB_ENV = {
    "USE_UNIFIED_PROVIDER": "",
    "ENABLE_BACKEND_ACCESS_CONTROL": "false",
    "DB_PROVIDER": "postgres",
    "VECTOR_DB_PROVIDER": "postgres",
    "GRAPH_DATABASE_PROVIDER": "postgres",
    "GRAPH_DATASET_DATABASE_HANDLER": "ladybug",
    "DB_HOST": "shared-host",
    "DB_PORT": "5432",
    "DB_USERNAME": "shared_user",
    "DB_PASSWORD": "shared_pass",
    "DB_NAME": "shared_db",
}

# Loggers that would emit a fallback/typo warning if the unification regressed.
DB_LOGGERS = {"PostgresConnectionResolver", "VectorEngine", "GraphEngine"}


def _clear_all_caches():
    get_relational_config.cache_clear()
    get_vectordb_config.cache_clear()
    get_graph_config.cache_clear()
    create_relational_engine.cache_clear()
    _create_vector_engine.cache_clear()
    _create_graph_engine.cache_clear()


@pytest.fixture(autouse=True)
def _clear_caches():
    _clear_all_caches()
    yield
    _clear_all_caches()


@pytest.fixture
def _fake_adapters():
    """Inject fake adapters (and a fake asyncpg) so engine construction succeeds
    without the postgres extra, and capture the connection URL each store builds.
    """
    captured = {}

    class FakePGVectorAdapter:
        def __init__(self, connection_string, api_key, embedding_engine):
            captured["vector"] = connection_string

    class FakePostgresAdapter:
        def __init__(self, connection_string):
            captured["graph"] = connection_string

    pgvector_module = types.ModuleType(
        "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter"
    )
    pgvector_module.PGVectorAdapter = FakePGVectorAdapter

    postgres_module = types.ModuleType("cognee.infrastructure.databases.graph.postgres.adapter")
    postgres_module.PostgresAdapter = FakePostgresAdapter

    with patch.dict(
        sys.modules,
        {
            "asyncpg": types.ModuleType("asyncpg"),
            "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter": pgvector_module,
            "cognee.infrastructure.databases.graph.postgres.adapter": postgres_module,
        },
    ):
        yield captured


def test_single_db_block_configures_all_three_stores(_fake_adapters, caplog):
    captured = _fake_adapters

    with patch.dict(os.environ, DB_ENV):
        _clear_all_caches()

        rel_params = get_relational_config().to_dict()
        vec_params = get_vectordb_config().to_dict()
        graph_params = get_graph_config().to_hashable_dict()

        with (
            patch(
                "cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter"
            ) as mock_sqlalchemy_adapter,
            patch(
                "cognee.infrastructure.databases.vector.create_vector_engine.get_embedding_engine",
                return_value=MagicMock(),
            ),
            caplog.at_level(logging.WARNING),
        ):
            create_relational_engine(**rel_params)
            create_vector_engine(**vec_params)
            create_graph_engine(**graph_params)

            captured["relational"] = mock_sqlalchemy_adapter.call_args[0][0]

    # All three stores resolve to the same single DB_* credential block.
    for label in ("relational", "vector", "graph"):
        url = captured[label]
        assert url.host == "shared-host", label
        assert url.port == 5432, label
        assert url.username == "shared_user", label
        assert url.password == "shared_pass", label
        assert url.database == "shared_db", label

    # No repetition needed -> no fallback warning and no typo warning.
    db_warnings = [
        r for r in caplog.records if r.levelno >= logging.WARNING and r.name in DB_LOGGERS
    ]
    assert db_warnings == [], [(r.name, r.getMessage()) for r in db_warnings]
