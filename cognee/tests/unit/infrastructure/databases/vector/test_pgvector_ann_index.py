from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, MetaData, String, Table


class _Embedder:
    def __init__(self, size=768):
        self.size = size

    def get_vector_size(self):
        return self.size

    async def embed_text(self, texts):
        return [[0.1] * self.size for _ in texts]


@pytest.mark.asyncio
async def test_ensure_vector_index_creates_hnsw_when_within_limit():
    pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    adapter = object.__new__(PGVectorAdapter)
    adapter.embedding_engine = _Embedder(size=768)

    mock_connection = AsyncMock()
    mock_begin = AsyncMock()
    mock_begin.__aenter__.return_value = mock_connection
    mock_begin.__aexit__.return_value = None

    adapter.engine = Mock()
    adapter.engine.begin.return_value = mock_begin

    await adapter._ensure_vector_index("DocumentChunk_text")

    mock_connection.execute.assert_awaited_once()
    sql_arg = str(mock_connection.execute.await_args[0][0])
    assert "CREATE INDEX IF NOT EXISTS" in sql_arg
    assert '"ix_DocumentChunk_text_vector_hnsw"' in sql_arg
    assert "USING hnsw (vector vector_cosine_ops)" in sql_arg


@pytest.mark.asyncio
async def test_ensure_vector_index_skips_when_exceeding_dimension_limit():
    pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    adapter = object.__new__(PGVectorAdapter)
    adapter.embedding_engine = _Embedder(size=3072)
    adapter.engine = Mock()

    await adapter._ensure_vector_index("DocumentChunk_text")

    adapter.engine.begin.assert_not_called()


@pytest.mark.asyncio
async def test_create_vector_index_ensures_hnsw_index():
    pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    adapter = object.__new__(PGVectorAdapter)
    adapter.create_collection = AsyncMock()
    adapter._ensure_vector_index = AsyncMock()

    await adapter.create_vector_index("DocumentChunk", "text")

    adapter.create_collection.assert_awaited_once_with("DocumentChunk_text")
    adapter._ensure_vector_index.assert_awaited_once_with("DocumentChunk_text")


@pytest.mark.asyncio
async def test_search_disables_indexscan_for_unbounded_limit():
    pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    adapter = object.__new__(PGVectorAdapter)
    adapter.embedding_engine = _Embedder(size=4)

    metadata = MetaData()
    test_table = Table(
        "test_col",
        metadata,
        Column("id", String, primary_key=True),
        Column("payload", JSON),
        Column("vector", Vector(4)),
    )
    adapter.get_table = AsyncMock(return_value=test_table)

    mock_count_result = Mock()
    mock_count_result.scalar_one.return_value = 100

    mock_items_result = Mock()
    mock_items_result.all.return_value = []

    mock_session = AsyncMock()
    # First execute is count query, second and third are SET LOCAL, fourth is search query
    mock_session.execute = AsyncMock(side_effect=[mock_count_result, None, None, mock_items_result])

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__.return_value = mock_session
    mock_session_ctx.__aexit__.return_value = None
    adapter.get_async_session = Mock(return_value=mock_session_ctx)

    results = await adapter.search(
        collection_name="test_col",
        query_vector=[0.1, 0.2, 0.3, 0.4],
        limit=None,
    )

    assert results == []
    # Verify SET LOCAL statements were executed to force sequential scan
    executed_statements = [
        str(call_args[0][0]) for call_args in mock_session.execute.await_args_list
    ]
    assert any("SET LOCAL enable_indexscan = off;" in s for s in executed_statements)
    assert any("SET LOCAL enable_bitmapscan = off;" in s for s in executed_statements)


@pytest.mark.asyncio
async def test_search_retains_indexscan_for_bounded_limit():
    pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    adapter = object.__new__(PGVectorAdapter)
    adapter.embedding_engine = _Embedder(size=4)

    metadata = MetaData()
    test_table = Table(
        "test_col",
        metadata,
        Column("id", String, primary_key=True),
        Column("payload", JSON),
        Column("vector", Vector(4)),
    )
    adapter.get_table = AsyncMock(return_value=test_table)

    mock_items_result = Mock()
    mock_items_result.all.return_value = []

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_items_result)

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__.return_value = mock_session
    mock_session_ctx.__aexit__.return_value = None
    adapter.get_async_session = Mock(return_value=mock_session_ctx)

    results = await adapter.search(
        collection_name="test_col",
        query_vector=[0.1, 0.2, 0.3, 0.4],
        limit=15,
    )

    assert results == []
    executed_statements = [
        str(call_args[0][0]) for call_args in mock_session.execute.await_args_list
    ]
    # Verify SET LOCAL was NOT executed when limit is bounded
    assert not any("SET LOCAL enable_indexscan" in s for s in executed_statements)
