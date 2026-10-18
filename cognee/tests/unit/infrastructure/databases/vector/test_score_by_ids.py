"""ID scoring uses exact requested rows, even when closer unrelated rows exist."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter
from cognee.infrastructure.databases.vector.turso.TursoVectorAdapter import TursoVectorAdapter


class _Embeddings:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        raise AssertionError("ID scoring must reuse the query and stored vectors")


class _Payload(BaseModel):
    text: str


@pytest.mark.asyncio
async def test_lancedb_scores_distant_ids_with_a_vector_index(tmp_path):
    from lancedb.index import IvfFlat

    adapter = LanceDBAdapter(str(tmp_path / "indexed"), None, _Embeddings())
    rows = [
        {"id": str(uuid4()), "vector": [1.0, float(i) / 100, 0.0], "payload": {"text": str(i)}}
        for i in range(128)
    ]
    distant = str(uuid4())
    rows.append({"id": distant, "vector": [-1.0, 0.0, 0.0], "payload": {"text": "distant"}})
    try:
        await adapter.upsert_raw_vectors("Indexed_text", rows, payload_schema=_Payload)
        table = await adapter.get_collection("Indexed_text")
        await table.create_index("vector", config=IvfFlat(distance_type="cosine", num_partitions=2))
        scores = await adapter.score_by_ids("Indexed_text", [distant], [1.0, 0.0, 0.0])
        assert [str(score.id) for score in scores] == [distant]
        assert scores[0].score == pytest.approx(2.0)
    finally:
        await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["lancedb", "lancedb-subprocess", "turso"])
async def test_scores_only_requested_rows_across_batches(tmp_path, backend):
    if backend == "turso":
        pytest.importorskip("turso")
        adapter = TursoVectorAdapter(str(tmp_path / "vectors.db"), None, _Embeddings())
        batch_target = (
            "cognee.infrastructure.databases.vector.turso.TursoVectorAdapter.QUERY_BATCH_SIZE"
        )
    else:
        adapter = LanceDBAdapter(str(tmp_path / "vectors"), None, _Embeddings())
        batch_target = (
            "cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter."
            "LanceDBAdapter.SCORE_ID_BATCH_SIZE"
        )
        if backend == "lancedb-subprocess":
            adapter = LanceDBAdapter.create_subprocess(
                str(tmp_path / "vectors"), None, _Embeddings()
            )
    collection = "RequestedVectors_text"
    first, second, closer, missing = [str(uuid4()) for _ in range(4)]
    try:
        rows = [
            {"id": first, "vector": [0.0, 1.0, 0.0], "payload": {"text": "first"}},
            {"id": second, "vector": [-1.0, 0.0, 0.0], "payload": {"text": "second"}},
            {"id": closer, "vector": [1.0, 0.0, 0.0], "payload": {"text": "unrelated"}},
        ]
        if backend == "turso":
            await adapter.create_collection(collection)
            for row in rows:
                await adapter._execute(
                    f'INSERT INTO "{collection}" (id, vector, payload) VALUES (?, vector32(?), ?)',
                    [row["id"], json.dumps(row["vector"]), json.dumps(row["payload"])],
                    commit=True,
                )
        else:
            await adapter.upsert_raw_vectors(collection, rows, payload_schema=_Payload)
        # Force several batches, including a duplicate ID and a missing row.
        with (
            patch(batch_target, 1),
            patch.object(
                adapter, "search", side_effect=AssertionError("must not run a collection search")
            ),
        ):
            scores = await adapter.score_by_ids(
                collection, [first, second, first, missing], [1.0, 0.0, 0.0]
            )
        assert len(scores) == 2
        assert {str(row.id): row.score for row in scores} == pytest.approx(
            {first: 1.0, second: 2.0}
        )
        assert all(row.payload is None for row in scores)
        assert await adapter.score_by_ids("MissingCollection", [], [1.0, 0.0, 0.0]) == []
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_lancedb_id_predicate_handles_quotes(tmp_path):
    adapter = LanceDBAdapter(str(tmp_path / "vectors"), None, _Embeddings())
    existing = str(uuid4())
    try:
        await adapter.upsert_raw_vectors(
            "QuotedIds_text",
            [{"id": existing, "vector": [1.0, 0.0, 0.0], "payload": {"text": "one"}}],
            payload_schema=_Payload,
        )
        scores = await adapter.score_by_ids(
            "QuotedIds_text", ["missing') OR true --"], [1.0, 0.0, 0.0]
        )
        assert scores == []
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_pgvector_scores_bound_ids_in_batches_without_payload():
    pytest.importorskip("pgvector")
    from pgvector.sqlalchemy import Vector
    from sqlalchemy import Column, MetaData, String, Table
    from sqlalchemy.dialects import postgresql

    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    table = Table("Entity_name", MetaData(), Column("id", String), Column("vector", Vector(3)))
    adapter = object.__new__(PGVectorAdapter)
    adapter.get_table = AsyncMock(return_value=table)
    captured = []
    first, second = str(uuid4()), str(uuid4())

    async def execute(statement):
        compiled = statement.compile(dialect=postgresql.dialect())
        captured.append(compiled)
        batch = compiled.params["id_1"]
        return SimpleNamespace(
            all=lambda: [SimpleNamespace(id=value, distance=0.25) for value in batch]
        )

    session = AsyncMock()
    session.execute.side_effect = execute
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    with (
        patch.object(adapter, "get_async_session", return_value=session_context),
        patch(
            "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter.QUERY_BATCH_SIZE", 1
        ),
    ):
        scores = await adapter.score_by_ids("Entity_name", [first, second, first], [1.0, 0.0, 0.0])
    assert [str(score.id) for score in scores] == [first, second]
    assert len(captured) == 2
    assert all(" WHERE " in str(query).replace("\n", " ") for query in captured)
    assert all("<=>" in str(query) for query in captured)
    assert all("payload" not in str(query) and "count(" not in str(query) for query in captured)
    assert await adapter.score_by_ids("MissingCollection", [], [1.0, 0.0, 0.0]) == []
