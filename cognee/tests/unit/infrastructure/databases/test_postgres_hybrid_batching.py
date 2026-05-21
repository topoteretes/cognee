"""Regression tests: PostgresHybridAdapter must respect embedding batch size.

The pghybrid adapter delegates embeddings to its underlying vector adapter.
Some embedding providers (e.g. gemini-embedding-001, batch limit 100) reject
calls with too many inputs in a single request. Both add_nodes_with_vectors
and add_edges_with_vectors must chunk by embedding_engine.get_batch_size()
before calling embed_data, mirroring index_data_points.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

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
async def test_add_edges_with_vectors_chunks_by_batch_size():
    """5 unique edge types + batch_size=2 → embed_data must be called 3 times."""
    adapter = _make_fake_hybrid(batch_size=2)
    edges = [(str(uuid4()), str(uuid4()), f"rel_{i}", {"edge_text": f"rel_{i}"}) for i in range(5)]

    await adapter.add_edges_with_vectors(edges)

    calls = adapter._vector.embed_data.await_args_list
    assert len(calls) == 3, f"expected 3 chunked calls, got {len(calls)}"
    all_texts = []
    for call in calls:
        (texts,) = call.args
        assert len(texts) <= 2, f"chunk size {len(texts)} exceeds batch_size 2"
        all_texts.extend(texts)
    assert sorted(all_texts) == sorted(f"rel_{i}" for i in range(5))


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
    assert len(calls) == 1, f"expected 1 fallback call, got {len(calls)}"
    (texts,) = calls[0].args
    assert sorted(texts) == sorted(f"rel_{i}" for i in range(5))


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
    assert len(calls) == 1, f"expected 1 fallback call, got {len(calls)}"
    (texts,) = calls[0].args
    assert sorted(texts) == sorted(f"rel_{i}" for i in range(5))
