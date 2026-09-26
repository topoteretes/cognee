from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from cognee.infrastructure.databases.vector.exceptions import SharedDatabasePruneError


class _Embedder:
    def get_vector_size(self):
        return 4


def _relational_database(database_name: str):
    return SimpleNamespace(
        db_uri=f"postgresql+asyncpg://cognee:cognee@localhost:5432/{database_name}",
        engine=SimpleNamespace(dialect=SimpleNamespace(name="postgresql")),
        sessionmaker=Mock(),
    )


def _adapter_configs():
    return (
        SimpleNamespace(pool_args=(), database_connect_args={}),
        SimpleNamespace(vector_pool_args=None),
    )


@pytest.mark.asyncio
async def test_distinct_vector_database_owns_its_engine():
    pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    relational_config, vector_config = _adapter_configs()
    with (
        patch(
            "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter.get_relational_engine",
            return_value=_relational_database("cognee"),
        ),
        patch(
            "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter.get_relational_config",
            return_value=relational_config,
        ),
        patch(
            "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter.get_vectordb_config",
            return_value=vector_config,
        ),
        patch(
            "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter.backend_access_control_enabled",
            return_value=False,
        ),
    ):
        adapter = PGVectorAdapter(
            "postgresql+asyncpg://cognee:cognee@localhost:5432/cognee_vectors",
            None,
            _Embedder(),
        )

    assert adapter._owns_engine is True
    await adapter.close()


def test_same_database_borrows_the_relational_engine():
    pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    relational = _relational_database("cognee")
    relational_config, vector_config = _adapter_configs()
    with (
        patch(
            "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter.get_relational_engine",
            return_value=relational,
        ),
        patch(
            "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter.get_relational_config",
            return_value=relational_config,
        ),
        patch(
            "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter.get_vectordb_config",
            return_value=vector_config,
        ),
        patch(
            "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter.backend_access_control_enabled",
            return_value=False,
        ),
    ):
        adapter = PGVectorAdapter(relational.db_uri, None, _Embedder())

    assert adapter._owns_engine is False
    assert adapter.engine is relational.engine
    assert adapter.sessionmaker is relational.sessionmaker


@pytest.mark.asyncio
async def test_prune_refuses_to_drop_a_shared_relational_database():
    pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    adapter = object.__new__(PGVectorAdapter)
    adapter._owns_engine = False
    adapter._metadata = Mock()
    adapter.delete_database = AsyncMock()

    with pytest.raises(SharedDatabasePruneError, match="shares the relational"):
        await adapter.prune()

    adapter._metadata.clear.assert_not_called()
    adapter.delete_database.assert_not_awaited()


@pytest.mark.asyncio
async def test_prune_still_drops_a_dedicated_vector_database():
    pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    adapter = object.__new__(PGVectorAdapter)
    adapter._owns_engine = True
    adapter._metadata = Mock()
    adapter.delete_database = AsyncMock()

    await adapter.prune()

    adapter._metadata.clear.assert_called_once_with()
    adapter.delete_database.assert_awaited_once_with()
