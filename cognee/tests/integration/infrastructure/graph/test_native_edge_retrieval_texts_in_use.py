"""Orphaned-EdgeType checks must not read the whole graph.

Every delete that removes an edge asks which of the removed edge retrieval texts
are still carried by a surviving edge. That used to be answered by
``get_graph_data()`` — every node and every edge, with properties — to decide
the fate of a handful of texts. Native adapters now answer in the store; the
full read survives only as the interface default for community adapters.

Each case checks the native answer against that default on the same data, so
the two cannot drift apart on the trim / fall-back-to-relationship-name rule.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.engine import DataPoint

logger = logging.getLogger(__name__)

try:
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    HAS_LADYBUG = True
except ModuleNotFoundError:
    HAS_LADYBUG = False

pytestmark = pytest.mark.asyncio


async def _make_neo4j_adapter():
    """Fresh (fully wiped) Neo4j graph adapter, or skip when ``.env`` isn't neo4j."""
    from cognee.infrastructure.databases.graph.config import get_graph_config

    config = get_graph_config()
    if config.graph_database_provider.lower() != "neo4j":
        pytest.skip("neo4j graph backend not configured (set GRAPH_DATABASE_PROVIDER=neo4j)")
    if not config.graph_database_url:
        pytest.skip("neo4j graph backend URL not configured")

    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

    adapter = Neo4jAdapter(
        graph_database_url=config.graph_database_url,
        graph_database_username=config.graph_database_username or None,
        graph_database_password=config.graph_database_password or None,
        graph_database_name=config.graph_database_name or None,
    )
    try:
        await adapter.initialize()
        await adapter.query("MATCH (n) DETACH DELETE n")
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("Ignoring exception in _make_neo4j_adapter", exc_info=True)
        await adapter.close()
        pytest.skip(f"neo4j graph backend not reachable: {exc}")
    return adapter


class _Ent(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


@pytest_asyncio.fixture(params=["ladybug", "neo4j"])
async def graph_adapter(request, tmp_path):
    if request.param == "ladybug":
        if not HAS_LADYBUG:
            pytest.skip("ladybug not installed")
        adapter = LadybugAdapter(str(tmp_path / "graph_db"))
    else:
        adapter = await _make_neo4j_adapter()

    try:
        yield adapter
    finally:
        await adapter.close()


async def _seed(adapter):
    alice, acme, bob, carol = (_Ent(id=uuid4(), name=name) for name in "ABCD")
    await adapter.add_nodes([alice, acme, bob, carol])
    a, b, c, d = (str(node.id) for node in (alice, acme, bob, carol))
    await adapter.add_edges(
        [
            # Stored edge_text wins over the relationship name.
            (a, b, "works_at", {"edge_text": "Alice works at Acme."}),
            # Stored edge_text is trimmed.
            (c, d, "knows", {"edge_text": "  Bob knows Carol.  "}),
            # No edge_text: the relationship name is the retrieval text.
            (a, c, "likes", {}),
            # Blank edge_text also falls back to the relationship name.
            (b, d, "manages", {"edge_text": "   "}),
        ]
    )


QUERIED = {
    "Alice works at Acme.",
    "Bob knows Carol.",
    "likes",
    "manages",
    "works_at",  # shadowed by that edge's edge_text, so not in use
    "Nobody carries this text.",
}
EXPECTED_IN_USE = {"Alice works at Acme.", "Bob knows Carol.", "likes", "manages"}


async def test_native_answer_matches_full_read_default(graph_adapter):
    await _seed(graph_adapter)

    full_read = await GraphDBInterface.get_edge_retrieval_texts_in_use(graph_adapter, QUERIED)
    assert full_read == EXPECTED_IN_USE

    graph_adapter.get_graph_data = AsyncMock(
        side_effect=AssertionError("unexpected full graph read")
    )
    assert await graph_adapter.get_edge_retrieval_texts_in_use(QUERIED) == EXPECTED_IN_USE
    graph_adapter.get_graph_data.assert_not_awaited()


async def test_empty_query_and_empty_graph(graph_adapter):
    assert await graph_adapter.get_edge_retrieval_texts_in_use(set()) == set()
    assert await graph_adapter.get_edge_retrieval_texts_in_use({"likes"}) == set()
