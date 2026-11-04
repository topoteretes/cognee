"""``search(document_name=...)`` keeps only the rows cut from the named documents.

Chunk rows carry ``document_name`` in their payload (the ``IndexSchema`` of every
built-in adapter), so a document scope is one predicate in the vector query, next
to the node-set predicate when both are given.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.engine import DataPoint

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        IndexSchema,
        LanceDBAdapter,
    )

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False

try:
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    HAS_PGVECTOR = True
except ModuleNotFoundError:
    HAS_PGVECTOR = False

try:
    # The adapter imports pyturso lazily, so the runtime itself decides availability.
    import turso

    from cognee.infrastructure.databases.vector.turso.TursoVectorAdapter import TursoVectorAdapter

    HAS_TURSO = True
except ModuleNotFoundError:
    HAS_TURSO = False


class _KeywordEmbeddingEngine:
    """apple -> x axis, banana -> y axis, so "apple" ranks doc_a's rows first."""

    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[1.0, 0.0, 0.1] if "apple" in text.lower() else [0.0, 1.0, 0.1] for text in texts]


class _ChunkRow(DataPoint):
    text: str
    document_name: str | None = None
    metadata: dict = {"index_fields": ["text"]}


def _rows():
    return [
        _ChunkRow(text="Apples are red fruit", document_name="doc_a", belongs_to_set=["fruit"]),
        _ChunkRow(text="Apples grow on trees", document_name="doc_a", belongs_to_set=[]),
        _ChunkRow(text="Bananas are yellow fruit", document_name="doc_b", belongs_to_set=["fruit"]),
        _ChunkRow(text="O'Neil's apple notes", document_name="o'neil", belongs_to_set=[]),
    ]


def _documents(results) -> list[str]:
    return sorted(result.payload["document_name"] for result in results)


# --- LanceDB: real rows -------------------------------------------------------


@pytest_asyncio.fixture
async def lancedb(tmp_path):
    if not HAS_LANCEDB:
        pytest.skip("lancedb extra is not installed")
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_KeywordEmbeddingEngine()
    )
    await adapter.create_collection("DocumentChunk_text", IndexSchema)
    await adapter.create_data_points(
        "DocumentChunk_text",
        [
            IndexSchema(
                id=str(uuid4()),
                **row.model_dump(include={"text", "document_name", "belongs_to_set"}),
            )
            for row in _rows()
        ],
    )
    return adapter


@pytest.mark.asyncio
async def test_lancedb_document_filter_keeps_only_the_named_documents(lancedb):
    results = await lancedb.search(
        "DocumentChunk_text",
        query_text="apple",
        limit=10,
        include_payload=True,
        document_name=["doc_a"],
    )
    assert _documents(results) == ["doc_a", "doc_a"]

    results = await lancedb.search(
        "DocumentChunk_text",
        query_text="apple",
        limit=10,
        include_payload=True,
        document_name=["doc_a", "doc_b"],
    )
    assert _documents(results) == ["doc_a", "doc_a", "doc_b"]


@pytest.mark.asyncio
async def test_lancedb_document_filter_escapes_quotes(lancedb):
    results = await lancedb.search(
        "DocumentChunk_text",
        query_text="apple",
        limit=10,
        include_payload=True,
        document_name=["o'neil"],
    )
    assert _documents(results) == ["o'neil"]


@pytest.mark.asyncio
async def test_lancedb_document_and_node_set_filters_combine(lancedb):
    results = await lancedb.search(
        "DocumentChunk_text",
        query_text="apple",
        limit=10,
        include_payload=True,
        node_name=["fruit"],
        document_name=["doc_a"],
    )
    assert [result.payload["text"] for result in results] == ["Apples are red fruit"]


@pytest.mark.asyncio
async def test_lancedb_unknown_document_matches_nothing(lancedb):
    results = await lancedb.search(
        "DocumentChunk_text",
        query_text="apple",
        limit=10,
        include_payload=True,
        document_name=["missing"],
    )
    assert results == []


@pytest.mark.asyncio
async def test_lancedb_without_a_document_filter_is_unscoped(lancedb):
    results = await lancedb.search(
        "DocumentChunk_text", query_text="apple", limit=10, include_payload=True
    )
    assert _documents(results) == ["doc_a", "doc_a", "doc_b", "o'neil"]


# --- PGVector: the statement -------------------------------------------------


class _RecordingQuery:
    def __init__(self):
        self.where_clauses = []
        self.order_by_clauses = []

    def where(self, clause):
        self.where_clauses.append(clause)
        return self

    def order_by(self, clause):
        self.order_by_clauses.append(clause)
        return self

    def limit(self, _limit):
        return self


class _RecordingSession:
    def __init__(self):
        self.statements = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, statement):
        self.statements.append(statement)
        return SimpleNamespace(all=list)


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_PGVECTOR, reason="pgvector extra is not installed")
async def test_pgvector_document_filter_is_a_where_clause_on_an_exact_scan(monkeypatch):
    from sqlalchemy import column

    import cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter as pg_module

    query = _RecordingQuery()
    monkeypatch.setattr(pg_module, "select", lambda *columns, **_kwargs: query)

    distance = SimpleNamespace(label=lambda _label: "distance_col")
    table = SimpleNamespace(
        c=SimpleNamespace(
            id="id_col",
            payload=column("payload"),
            vector=SimpleNamespace(cosine_distance=lambda _vector: distance),
        )
    )
    session = _RecordingSession()
    adapter = PGVectorAdapter.__new__(PGVectorAdapter)
    adapter.embedding_engine = _KeywordEmbeddingEngine()
    # A pgvector that can index the column: the filter must still take the exact scan.
    adapter._pgvector_version = AsyncMock(return_value=(0, 8, 0))
    adapter.get_table = AsyncMock(return_value=table)
    adapter.get_async_session = lambda: session

    await PGVectorAdapter.search(
        adapter,
        collection_name="DocumentChunk_text",
        query_vector=[0.1, 0.2, 0.3],
        limit=5,
        include_payload=True,
        document_name=["doc_a", "doc_b"],
    )

    [clause] = query.where_clauses
    rendered = str(clause.compile(compile_kwargs={"literal_binds": True}))
    assert "->> 'document_name'" in rendered
    assert "IN ('doc_a', 'doc_b')" in rendered
    assert query.order_by_clauses == ["similarity"]
    executed = [str(statement) for statement in session.statements]
    assert any("enable_indexscan = off" in statement for statement in executed)
    assert not any("enable_seqscan = off" in statement for statement in executed)


# --- Turso: real rows ---------------------------------------------------------


@pytest_asyncio.fixture
async def turso(tmp_path):
    if not HAS_TURSO:
        pytest.skip("pyturso not installed")
    adapter = TursoVectorAdapter(
        url=str(tmp_path / "turso.db"), api_key=None, embedding_engine=_KeywordEmbeddingEngine()
    )
    try:
        await adapter.create_data_points("DocumentChunk_text", _rows())
        yield adapter
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_turso_document_filter_keeps_only_the_named_documents(turso):
    results = await turso.search(
        "DocumentChunk_text",
        query_text="apple",
        limit=None,
        include_payload=True,
        document_name=["doc_a"],
    )
    assert _documents(results) == ["doc_a", "doc_a"]

    results = await turso.search(
        "DocumentChunk_text",
        query_text="apple",
        limit=None,
        include_payload=True,
        document_name=["o'neil"],
    )
    assert _documents(results) == ["o'neil"]


@pytest.mark.asyncio
async def test_turso_document_and_node_set_filters_combine(turso):
    results = await turso.search(
        "DocumentChunk_text",
        query_text="apple",
        limit=None,
        include_payload=True,
        node_name=["fruit"],
        node_name_filter_operator="AND",
        document_name=["doc_a", "doc_b"],
    )
    assert sorted(result.payload["text"] for result in results) == [
        "Apples are red fruit",
        "Bananas are yellow fruit",
    ]


@pytest.mark.asyncio
async def test_turso_unknown_document_matches_nothing(turso):
    results = await turso.search(
        "DocumentChunk_text",
        query_text="apple",
        limit=None,
        include_payload=True,
        document_name=["missing"],
    )
    assert results == []
