"""Regression tests: PostgresHybridAdapter must respect embedding batch size.

The pghybrid adapter delegates embeddings to its underlying vector adapter.
Some embedding providers (e.g. gemini-embedding-001, batch limit 100) reject
calls with too many inputs in a single request. Both add_nodes_with_vectors
and add_edges_with_vectors must chunk by embedding_engine.get_batch_size()
before calling embed_data, mirroring index_data_points.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
import json

import pytest

pytest.importorskip("asyncpg", reason="PostgresHybridAdapter requires the postgres extra")
pytest.importorskip("pgvector", reason="PostgresHybridAdapter requires the postgres extra")

from cognee.infrastructure.engine import DataPoint  # noqa: E402
from cognee.infrastructure.databases.hybrid.postgres.adapter import (  # noqa: E402
    PostgresHybridAdapter,
)


class _Node(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class _SummaryNode(DataPoint):
    text: str
    source_chunk_id: str
    metadata: dict = {"index_fields": ["text"]}


def _make_fake_hybrid(batch_size: int):
    """Build a PostgresHybridAdapter with stubbed graph/vector adapters.

    embed_data raises if called with more inputs than batch_size, so any
    failure to chunk surfaces as a ValueError from the test.
    """
    fake = PostgresHybridAdapter.__new__(PostgresHybridAdapter)

    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    fake._graph = MagicMock()
    fake._graph.initialize = AsyncMock()
    fake._graph._session = MagicMock(return_value=session_cm)

    async def embed_data(texts):
        # batch_size <= 0 means "no per-call cap" — the production code is expected
        # to fall back to a single all-in-one call, so don't enforce here.
        if batch_size > 0 and len(texts) > batch_size:
            raise ValueError(
                f"embed_data called with {len(texts)} texts, exceeds batch_size={batch_size}"
            )
        return [[0.1, 0.2] for _ in texts]

    fake._vector = MagicMock()
    fake._vector.embed_data = AsyncMock(side_effect=embed_data)
    fake._vector.create_vector_index = AsyncMock()
    fake._vector.embedding_engine = MagicMock()
    fake._vector.embedding_engine.get_batch_size = MagicMock(return_value=batch_size)
    fake.embedding_engine = fake._vector.embedding_engine

    return fake


def _session_from_fake_hybrid(adapter):
    return adapter._graph._session.return_value.__aenter__.return_value


@pytest.mark.asyncio
async def test_add_nodes_with_vectors_chunks_by_batch_size():
    """5 nodes + batch_size=2 → embed_data must be called 3 times in chunks of ≤2."""
    adapter = _make_fake_hybrid(batch_size=2)
    nodes = [_Node(id=uuid4(), name=f"n{i}") for i in range(5)]

    await adapter.add_nodes_with_vectors(nodes)

    calls = adapter._vector.embed_data.await_args_list
    assert len(calls) == 3, f"expected 3 chunked calls, got {len(calls)}"
    all_texts = []
    for call in calls:
        (texts,) = call.args
        assert len(texts) <= 2, f"chunk size {len(texts)} exceeds batch_size 2"
        all_texts.extend(texts)
    assert sorted(all_texts) == sorted(f"n{i}" for i in range(5))


@pytest.mark.asyncio
async def test_add_nodes_with_vectors_preserves_summary_source_chunk_id_in_payload():
    adapter = _make_fake_hybrid(batch_size=10)
    source_chunk_id = str(uuid4())
    node = _SummaryNode(text="summary", source_chunk_id=source_chunk_id, importance_weight=0.9)

    await adapter.add_nodes_with_vectors([node])

    session = _session_from_fake_hybrid(adapter)
    vector_rows = session.execute.await_args_list[1].args[1]
    payload = json.loads(vector_rows[0]["payload"])
    assert payload["source_chunk_id"] == source_chunk_id
    assert payload["importance_weight"] == 0.9


@pytest.mark.asyncio
async def test_add_edges_with_vectors_chunks_by_batch_size():
    """Type names and instance prose are each chunked by embedding batch size."""
    adapter = _make_fake_hybrid(batch_size=2)
    edges = [(str(uuid4()), str(uuid4()), f"rel_{i}", {"edge_text": f"rel_{i}"}) for i in range(5)]

    await adapter.add_edges_with_vectors(edges)

    calls = adapter._vector.embed_data.await_args_list
    assert len(calls) == 6, f"expected 6 chunked calls, got {len(calls)}"
    all_texts = []
    for call in calls:
        (texts,) = call.args
        assert len(texts) <= 2, f"chunk size {len(texts)} exceeds batch_size 2"
        all_texts.extend(texts)
    assert sorted(all_texts) == sorted([f"rel_{i}" for i in range(5)] * 2)


@pytest.mark.asyncio
async def test_add_edges_with_vectors_falls_back_from_blank_edge_text_to_relationship_name():
    adapter = _make_fake_hybrid(batch_size=10)
    edges = [
        (str(uuid4()), str(uuid4()), "blank_rel", {"edge_text": "   "}),
        (str(uuid4()), str(uuid4()), "none_rel", {"edge_text": None}),
    ]

    await adapter.add_edges_with_vectors(edges)

    embedded_texts = [
        text for call in adapter._vector.embed_data.await_args_list for text in call.args[0]
    ]
    assert sorted(embedded_texts) == ["blank_rel", "blank_rel", "none_rel", "none_rel"]


@pytest.mark.asyncio
async def test_add_nodes_with_vectors_batch_size_zero_falls_back_to_single_call():
    """get_batch_size()==0 must not crash on `range() arg 3 must not be zero`.

    The adapter falls back to len(texts) so the loop runs exactly once and
    embed_data is called with all inputs at once (matches pre-batching behavior).
    """
    adapter = _make_fake_hybrid(batch_size=0)
    nodes = [_Node(id=uuid4(), name=f"n{i}") for i in range(5)]

    await adapter.add_nodes_with_vectors(nodes)

    calls = adapter._vector.embed_data.await_args_list
    assert len(calls) == 1, f"expected 1 fallback call, got {len(calls)}"
    (texts,) = calls[0].args
    assert sorted(texts) == sorted(f"n{i}" for i in range(5))


@pytest.mark.asyncio
async def test_add_edges_with_vectors_batch_size_zero_falls_back_to_single_call():
    """Same fallback for edges: get_batch_size()==0 → one all-inputs call."""
    adapter = _make_fake_hybrid(batch_size=0)
    edges = [(str(uuid4()), str(uuid4()), f"rel_{i}", {"edge_text": f"rel_{i}"}) for i in range(5)]

    await adapter.add_edges_with_vectors(edges)

    calls = adapter._vector.embed_data.await_args_list
    assert len(calls) == 2, f"expected 2 fallback calls, got {len(calls)}"
    assert sorted(calls[0].args[0]) == sorted(f"rel_{i}" for i in range(5))
    assert sorted(calls[1].args[0]) == sorted(f"rel_{i}" for i in range(5))


@pytest.mark.asyncio
async def test_add_nodes_with_vectors_negative_batch_size_falls_back_to_single_call():
    """Negative get_batch_size() must not silently drop embeddings.

    range(0, n, -1) yields no iterations, so without the guard the loop would
    skip every input and `vectors` would stay empty — silent data loss. The
    fallback to len(items) makes the loop run once with all inputs.
    """
    adapter = _make_fake_hybrid(batch_size=-1)
    nodes = [_Node(id=uuid4(), name=f"n{i}") for i in range(5)]

    await adapter.add_nodes_with_vectors(nodes)

    calls = adapter._vector.embed_data.await_args_list
    assert len(calls) == 1, f"expected 1 fallback call, got {len(calls)}"
    (texts,) = calls[0].args
    assert sorted(texts) == sorted(f"n{i}" for i in range(5))


@pytest.mark.asyncio
async def test_add_edges_with_vectors_negative_batch_size_falls_back_to_single_call():
    """Same fallback for edges: negative get_batch_size() → one all-inputs call."""
    adapter = _make_fake_hybrid(batch_size=-1)
    edges = [(str(uuid4()), str(uuid4()), f"rel_{i}", {"edge_text": f"rel_{i}"}) for i in range(5)]

    await adapter.add_edges_with_vectors(edges)

    calls = adapter._vector.embed_data.await_args_list
    assert len(calls) == 2, f"expected 2 fallback calls, got {len(calls)}"
    assert sorted(calls[0].args[0]) == sorted(f"rel_{i}" for i in range(5))
    assert sorted(calls[1].args[0]) == sorted(f"rel_{i}" for i in range(5))


@pytest.mark.asyncio
async def test_add_edges_with_vectors_indexes_relationship_types_and_edge_prose_separately():
    """Type payloads use graph-wide counts; prose rows retain shared edge object ids."""
    adapter = _make_fake_hybrid(batch_size=10)
    first_id, second_id = str(uuid4()), str(uuid4())
    edges = [
        (
            "a",
            "b",
            "depends_on",
            {"edge_object_id": first_id, "edge_text": "Package A depends on Package B."},
        ),
        (
            "c",
            "d",
            "depends_on",
            {"edge_object_id": second_id, "edge_text": "Service C depends on Service D."},
        ),
    ]
    session = _session_from_fake_hybrid(adapter)
    count_result = MagicMock()
    count_result.fetchall.return_value = [("depends_on", 7)]
    session.execute.side_effect = [MagicMock(), count_result, MagicMock(), MagicMock()]

    await adapter.add_edges_with_vectors(edges)

    embedded_type_texts = adapter._vector.embed_data.await_args_list[0].args[0]
    embedded_instance_texts = adapter._vector.embed_data.await_args_list[1].args[0]
    assert embedded_type_texts == ["depends_on"]
    assert sorted(embedded_instance_texts) == [
        "Package A depends on Package B.",
        "Service C depends on Service D.",
    ]
    assert adapter._vector.create_vector_index.await_args_list[0].args == (
        "EdgeType",
        "relationship_name",
    )
    assert adapter._vector.create_vector_index.await_args_list[1].args == ("EdgeInstance", "text")

    calls_by_table = {
        str(call.args[0].text).split("INSERT INTO ", 1)[1].split(" ", 1)[0]: call.args[1]
        for call in session.execute.await_args_list
        if 'INSERT INTO "Edge' in str(call.args[0].text)
    }
    type_rows = calls_by_table['"EdgeType_relationship_name"']
    instance_rows = calls_by_table['"EdgeInstance_text"']
    type_payload = json.loads(type_rows[0]["payload"])
    assert type_payload["number_of_edges"] == 7
    assert {row["id"] for row in instance_rows} == {first_id, second_id}
    assert session.commit.await_count == 1


@pytest.mark.asyncio
async def test_add_edges_with_vectors_rewrites_instance_vector_on_same_edge_id():
    """A repeat write updates both payload and vector for the existing instance id."""
    adapter = _make_fake_hybrid(batch_size=10)
    edge_id = str(uuid4())
    edge = ("a", "b", "depends_on", {"edge_object_id": edge_id, "edge_text": "first prose"})
    session = _session_from_fake_hybrid(adapter)
    count_result = MagicMock()
    count_result.fetchall.return_value = [("depends_on", 1)]
    session.execute.side_effect = [
        MagicMock(),
        count_result,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        count_result,
        MagicMock(),
        MagicMock(),
    ]
    adapter._vector.embed_data.side_effect = [
        [[0.1, 0.2]],
        [[0.3, 0.4]],
        [[0.1, 0.2]],
        [[0.5, 0.6]],
    ]

    await adapter.add_edges_with_vectors([edge])
    await adapter.add_edges_with_vectors(
        [("a", "b", "depends_on", {"edge_object_id": edge_id, "edge_text": "replacement prose"})]
    )

    instance_insert_calls = [
        call
        for call in session.execute.await_args_list
        if 'INSERT INTO "EdgeInstance_text"' in str(call.args[0].text)
    ]
    assert len(instance_insert_calls) == 2
    assert json.loads(instance_insert_calls[1].args[1][0]["payload"])["text"] == "replacement prose"
    assert instance_insert_calls[1].args[1][0]["vector"] == "[0.5, 0.6]"
    assert "vector = EXCLUDED.vector" in str(instance_insert_calls[1].args[0].text)
