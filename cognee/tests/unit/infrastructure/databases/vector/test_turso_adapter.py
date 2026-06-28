"""Behavioral tests for ``TursoVectorAdapter`` against a real embedded libSQL DB.

These run without an LLM/embedding provider: a deterministic fake embedding
engine maps keywords to vector axes so cosine ordering and NodeSet filtering
are exercised on real ``vector_distance_cos`` / JSON1 SQL, not mocks.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

try:
    import libsql_experimental  # noqa: F401

    from cognee.infrastructure.databases.vector.turso.TursoVectorAdapter import (
        TursoVectorAdapter,
    )
    from cognee.infrastructure.engine import DataPoint

    HAS_LIBSQL = True
except ModuleNotFoundError:
    HAS_LIBSQL = False


pytestmark = pytest.mark.skipif(not HAS_LIBSQL, reason="libsql-experimental not installed")

DIM = 4


class _FakeEmbeddingEngine:
    """Deterministic embeddings: keyword -> axis, so distances are meaningful."""

    def get_vector_size(self):
        return DIM

    async def embed_text(self, texts):
        out = []
        for text in texts:
            lowered = text.lower()
            vector = [
                1.0 if "quantum" in lowered else 0.0,
                1.0 if "nlp" in lowered or "language" in lowered else 0.0,
                1.0 if "alice" in lowered else 0.0,
                0.1,
            ]
            if sum(vector[:3]) == 0:
                vector[3] += 0.5
            out.append(vector)
        return out


if HAS_LIBSQL:

    class _Doc(DataPoint):
        text: str
        metadata: dict = {"index_fields": ["text"]}


@pytest_asyncio.fixture
async def adapter(tmp_path):
    db_path = str(tmp_path / "turso_test.db")
    instance = TursoVectorAdapter(
        url=db_path, api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    try:
        yield instance
    finally:
        await instance.close()


def _docs():
    return [
        _Doc(text="quantum computers are fast", belongs_to_set=["Quantum", "Computers"]),
        _Doc(text="natural language processing nlp", belongs_to_set=["NLP"]),
        _Doc(text="alice studies quantum", belongs_to_set=["Quantum"]),
    ]


@pytest.mark.asyncio
async def test_has_collection_lifecycle(adapter):
    assert await adapter.has_collection("Entity_name") is False
    await adapter.create_collection("Entity_name")
    assert await adapter.has_collection("Entity_name") is True


@pytest.mark.asyncio
async def test_create_data_points_and_search_ordering(adapter):
    docs = _docs()
    await adapter.create_data_points("DocumentChunk_text", docs)

    results = await adapter.search(
        "DocumentChunk_text", query_text="quantum", limit=3, include_payload=True
    )
    assert len(results) == 3
    # cosine distance is lower-is-better; results must be ascending by score
    assert results == sorted(results, key=lambda r: r.score)
    assert "quantum" in results[0].payload["text"]


@pytest.mark.asyncio
async def test_search_none_limit_returns_all(adapter):
    await adapter.create_data_points("DocumentChunk_text", _docs())
    results = await adapter.search("DocumentChunk_text", query_text="quantum", limit=None)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_nodeset_or_filtering(adapter):
    await adapter.create_data_points("DocumentChunk_text", _docs())
    results = await adapter.search(
        "DocumentChunk_text",
        query_text="quantum",
        limit=None,
        include_payload=True,
        node_name=["NLP", "Quantum"],
        node_name_filter_operator="OR",
    )
    returned_tags = {tag for r in results for tag in r.payload["belongs_to_set"]}
    assert returned_tags <= {"NLP", "Quantum", "Computers"}
    assert any("nlp" in r.payload["text"] for r in results)


@pytest.mark.asyncio
async def test_nodeset_and_filtering(adapter):
    await adapter.create_data_points("DocumentChunk_text", _docs())
    results = await adapter.search(
        "DocumentChunk_text",
        query_text="quantum",
        limit=None,
        include_payload=True,
        node_name=["Quantum", "Computers"],
        node_name_filter_operator="AND",
    )
    assert len(results) == 1
    assert "quantum computers" in results[0].payload["text"]


@pytest.mark.asyncio
async def test_retrieve_by_ids(adapter):
    docs = _docs()
    await adapter.create_data_points("DocumentChunk_text", docs)
    got = await adapter.retrieve("DocumentChunk_text", [d.id for d in docs])
    assert {str(r.id) for r in got} == {str(d.id) for d in docs}


@pytest.mark.asyncio
async def test_belongs_to_set_merge_on_conflict(adapter):
    docs = _docs()
    await adapter.create_data_points("DocumentChunk_text", docs)

    duplicate = _Doc(text="alice studies quantum", belongs_to_set=["Mechanics"])
    duplicate.id = docs[2].id
    await adapter.create_data_points("DocumentChunk_text", [duplicate])

    merged = (await adapter.retrieve("DocumentChunk_text", [docs[2].id]))[0]
    assert set(merged.payload["belongs_to_set"]) == {"Quantum", "Mechanics"}


@pytest.mark.asyncio
async def test_remove_belongs_to_set_tags(adapter):
    docs = _docs()
    await adapter.create_data_points("DocumentChunk_text", docs)

    # Stripping one of several tags keeps the row.
    await adapter.remove_belongs_to_set_tags(["Computers"])
    kept = (await adapter.retrieve("DocumentChunk_text", [docs[0].id]))[0]
    assert set(kept.payload["belongs_to_set"]) == {"Quantum"}

    # Stripping the last tag empties the array, so the row is deleted.
    await adapter.remove_belongs_to_set_tags(["NLP"])
    assert await adapter.retrieve("DocumentChunk_text", [docs[1].id]) == []


@pytest.mark.asyncio
async def test_remove_tags_preserves_untagged_rows(adapter):
    # Regression: a row stored with an empty belongs_to_set (e.g. an untagged
    # index row) must NOT be deleted when an unrelated tag is removed. The detag
    # must only touch rows that actually contained one of the removed tags.
    tagged = _Doc(text="quantum tagged", belongs_to_set=["Quantum"])
    untagged = _Doc(text="untagged chunk", belongs_to_set=[])
    await adapter.create_data_points("DocumentChunk_text", [tagged, untagged])

    await adapter.remove_belongs_to_set_tags(["Quantum"])

    # tagged row had only Quantum -> emptied -> deleted
    assert await adapter.retrieve("DocumentChunk_text", [tagged.id]) == []
    # untagged row never had Quantum -> must survive
    assert len(await adapter.retrieve("DocumentChunk_text", [untagged.id])) == 1


@pytest.mark.asyncio
async def test_delete_data_points(adapter):
    docs = _docs()
    await adapter.create_data_points("DocumentChunk_text", docs)
    await adapter.delete_data_points("DocumentChunk_text", [docs[0].id])
    assert await adapter.retrieve("DocumentChunk_text", [docs[0].id]) == []


@pytest.mark.asyncio
async def test_batch_search(adapter):
    await adapter.create_data_points("DocumentChunk_text", _docs())
    batches = await adapter.batch_search(
        "DocumentChunk_text", ["quantum", "alice"], limit=2, include_payload=True
    )
    assert len(batches) == 2
    assert all(isinstance(batch, list) for batch in batches)


@pytest.mark.asyncio
async def test_prune_drops_all_collections(adapter):
    await adapter.create_data_points("DocumentChunk_text", _docs())
    await adapter.prune()
    assert await adapter.has_collection("DocumentChunk_text") is False
    assert await adapter.get_table_names() == []
