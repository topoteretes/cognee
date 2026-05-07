"""Regression tests: PostgresHybridAdapter must build collection keys
losslessly and tolerate DataPoints with no metadata.

Both bugs were flagged by coderabbitai on PR #2587 against the same lines
of ``add_nodes_with_vectors`` and remained unfixed:

1. The collection key was constructed as ``f"{type_name}_{field_name}"``
   then later recovered with ``rsplit("_", 1)``. For multi-underscore
   field names (e.g. ``source_code``) this corrupts the recovered tuple
   to ``("CodeGraphEntity_source", "code")``, creating the wrong vector
   table and silently misrouting search.

2. ``dp.metadata.get("index_fields", [])`` had no None guard. ``metadata``
   is Optional on ``DataPoint`` and any DataPoint with ``metadata=None``
   crashed with ``AttributeError``. The sibling ``index_data_points`` task
   already guards with the same ``hasattr/truthy`` check used here.
"""

from typing import Optional
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("asyncpg", reason="PostgresHybridAdapter requires the postgres extra")
pytest.importorskip("pgvector", reason="PostgresHybridAdapter requires the postgres extra")

from cognee.infrastructure.engine import DataPoint  # noqa: E402
from cognee.infrastructure.databases.hybrid.postgres.adapter import (  # noqa: E402
    PostgresHybridAdapter,
)


def _make_fake_hybrid():
    """Build a PostgresHybridAdapter with stubbed graph/vector adapters.

    Same shape as the helper in ``test_postgres_hybrid_batching.py`` but
    with a generous batch size (chunking is not under test here) and an
    embed_data stub that always returns a fixed vector.
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
    fake._graph_session = session

    async def embed_data(texts):
        return [[0.1, 0.2] for _ in texts]

    fake._vector = MagicMock()
    fake._vector.embed_data = AsyncMock(side_effect=embed_data)
    fake._vector.create_vector_index = AsyncMock()
    fake._vector.embedding_engine = MagicMock()
    fake._vector.embedding_engine.get_batch_size = MagicMock(return_value=100)
    fake.embedding_engine = fake._vector.embedding_engine

    return fake


class _SingleNode(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class _CodeGraphEntity(DataPoint):
    """DataPoint with a multi-underscore index field — the regression case."""

    source_code: str
    metadata: dict = {"index_fields": ["source_code"]}


class _OptionalMetadataNode(DataPoint):
    """DataPoint that allows ``metadata=None`` past pydantic validation."""

    name: str
    metadata: Optional[dict] = None


@pytest.mark.asyncio
async def test_single_underscore_field_creates_correct_index():
    """Sanity check: a simple single-word field still resolves to the right index."""
    adapter = _make_fake_hybrid()
    nodes = [_SingleNode(id=uuid4(), name=f"n{i}") for i in range(3)]

    await adapter.add_nodes_with_vectors(nodes)

    create_calls = adapter._vector.create_vector_index.await_args_list
    assert len(create_calls) == 1, f"expected 1 index, got {len(create_calls)}"
    assert create_calls[0].args == ("_SingleNode", "name")


@pytest.mark.asyncio
async def test_multi_underscore_field_preserves_full_name():
    """Bug #5 regression: a field like ``source_code`` must NOT be split as
    ``("..._source", "code")``. The whole field name has to survive intact."""
    adapter = _make_fake_hybrid()
    entities = [_CodeGraphEntity(id=uuid4(), source_code=f"def f{i}(): pass") for i in range(3)]

    await adapter.add_nodes_with_vectors(entities)

    create_calls = adapter._vector.create_vector_index.await_args_list
    assert len(create_calls) == 1, f"expected 1 index, got {len(create_calls)}"
    type_arg, field_arg = create_calls[0].args
    assert (type_arg, field_arg) == ("_CodeGraphEntity", "source_code"), (
        f"expected ('_CodeGraphEntity', 'source_code'), got ({type_arg!r}, {field_arg!r}) — "
        "rsplit('_', 1) corrupted the multi-underscore field name"
    )


@pytest.mark.asyncio
async def test_metadata_none_does_not_crash():
    """Bug #6 regression: a DataPoint with ``metadata=None`` must not raise
    ``AttributeError``. The graph row should still be inserted; no vector
    index should be created for that point. Mirrors the silently-skip
    semantics in ``index_data_points``."""
    adapter = _make_fake_hybrid()
    node = _OptionalMetadataNode(id=uuid4(), name="ghost")
    assert node.metadata is None  # construct-time guard for the precondition

    await adapter.add_nodes_with_vectors([node])

    # No vector index attempted for the unindexable point.
    assert adapter._vector.create_vector_index.await_args_list == []
    # But the graph_node insert still ran (the node row is preserved).
    executed = adapter._graph_session.execute.await_args_list
    assert any("graph_node" in str(call.args[0]) for call in executed), (
        "expected graph_node INSERT to run even when metadata is None"
    )


@pytest.mark.asyncio
async def test_collection_key_round_trip_is_backwards_compatible():
    """Backwards-compat invariant: the pre-fix lossy ``rsplit("_", 1)``
    round-trip produced the SAME final collection name as the post-fix
    tuple-keyed code, because ``PGVectorAdapter.create_vector_index``
    immediately re-joins its two arguments with the same separator —
    f-string concatenation across an underscore is associative.

    This means the rsplit cleanup in ``add_nodes_with_vectors`` is a
    latent-footgun fix, not a live behavior change: existing pgvector
    tables persist with their pre-fix names, so no data migration is
    required. This test pins that invariant — if anyone changes
    ``PGVectorAdapter.create_vector_index`` in a way that breaks the
    associativity (e.g. uses the args separately for routing), this
    test fails and surfaces that the cleanup is no longer a no-op.
    """
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
        PGVectorAdapter,
    )

    fake = PGVectorAdapter.__new__(PGVectorAdapter)
    fake.create_collection = AsyncMock()

    # Pre-fix call shape: rsplit("_", 1) on "CodeGraphEntity_source_code"
    # yields ("CodeGraphEntity_source", "code"). This is what the buggy
    # code passed to create_vector_index.
    await fake.create_vector_index("CodeGraphEntity_source", "code")
    pre_fix_name = fake.create_collection.await_args_list[-1].args[0]

    # Post-fix call shape: tuple key (type_name, field_name) preserved as
    # ("CodeGraphEntity", "source_code"). This is what the fixed code passes.
    await fake.create_vector_index("CodeGraphEntity", "source_code")
    post_fix_name = fake.create_collection.await_args_list[-1].args[0]

    assert pre_fix_name == post_fix_name == "CodeGraphEntity_source_code", (
        f"backwards-compat invariant broken: pre-fix produced {pre_fix_name!r}, "
        f"post-fix produced {post_fix_name!r}. If this fails, the rsplit cleanup "
        "in add_nodes_with_vectors is no longer a no-op and existing pgvector "
        "tables may be stranded."
    )
