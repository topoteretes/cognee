"""Unit tests for relationship-type quoting and batch fallback in ``NeptuneGraphDB``.

openCypher cannot parameterize a created relationship type, so the type is interpolated
into the query string. It must be backtick-quoted (and internal backticks doubled) so that
labels produced by LLM extraction -- which routinely contain spaces, hyphens, or other
characters -- do not break the query or allow injection.

Skipped automatically when the Neptune optional dependencies (langchain_aws, botocore) are
not installed.
"""

import pytest
from unittest.mock import AsyncMock

pytest.importorskip("langchain_aws", reason="Neptune tests require langchain_aws")
pytest.importorskip("botocore", reason="Neptune tests require botocore")

from cognee.infrastructure.databases.graph.neptune_driver.adapter import (  # noqa: E402
    NeptuneGraphDB,
)


class _FakeGraphDB:
    """Binds the methods under test onto a stand-in with a mocked ``query()``."""

    add_edge = NeptuneGraphDB.add_edge
    add_edges = NeptuneGraphDB.add_edges
    _GRAPH_NODE_LABEL = NeptuneGraphDB._GRAPH_NODE_LABEL

    def __init__(self):
        self.query = AsyncMock(return_value=[])
        self._serialize_properties = lambda properties: properties


def _last_query(fake) -> str:
    return fake.query.call_args.args[0]


@pytest.mark.asyncio
async def test_add_edge_quotes_relationship_with_space():
    fake = _FakeGraphDB()
    await fake.add_edge("src", "tgt", "works at", {})
    query = _last_query(fake)
    assert "[r:`works at`]" in query
    assert "[r:works at]" not in query


@pytest.mark.asyncio
async def test_add_edge_escapes_internal_backtick():
    fake = _FakeGraphDB()
    await fake.add_edge("src", "tgt", "weird`name", {})
    query = _last_query(fake)
    assert "[r:`weird``name`]" in query


@pytest.mark.asyncio
async def test_add_edges_batch_quotes_relationship_with_space():
    fake = _FakeGraphDB()
    await fake.add_edges([("a", "b", "located in", {})])
    query = _last_query(fake)
    assert "[r:`located in`]" in query
    assert "[r:located in]" not in query


@pytest.mark.asyncio
async def test_add_edges_fallback_retries_real_edges_on_batch_failure():
    fake = _FakeGraphDB()
    # The batch query fails, so add_edges falls back to per-edge creation.
    fake.query = AsyncMock(side_effect=Exception("boom"))
    fake.add_edge = AsyncMock()
    edges = [("a", "b", "REL", {"k": "v"}), ("c", "d", "REL", {})]
    await fake.add_edges(edges)
    # The fallback must retry the real edge tuples, not characters of the relationship name.
    fake.add_edge.assert_any_await("a", "b", "REL", {"k": "v"})
    fake.add_edge.assert_any_await("c", "d", "REL", {})
    assert fake.add_edge.await_count == 2
