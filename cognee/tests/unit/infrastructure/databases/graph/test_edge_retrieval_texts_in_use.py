"""The inherited full-read default of ``get_edge_retrieval_texts_in_use``.

Ladybug and Neo4j answer in the store (see the integration test of the same
name); community adapters inherit this default, so it keeps a test of its own.
"""

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface


class _FullReadAdapter:
    """Not a GraphDBInterface subclass: see test_top_degree_node_ids for why."""

    def __init__(self, edges):
        self._edges = edges
        self.full_reads = 0

    async def get_graph_data(self):
        self.full_reads += 1
        return [], self._edges

    async def get_edge_retrieval_texts_in_use(self, edge_texts):
        return await GraphDBInterface.get_edge_retrieval_texts_in_use(self, edge_texts)


@pytest.mark.asyncio
async def test_default_applies_edge_text_then_relationship_name():
    adapter = _FullReadAdapter(
        [
            ("a", "b", "works_at", {"edge_text": "Alice works at Acme."}),
            ("c", "d", "knows", {"edge_text": "  Bob knows Carol.  "}),
            ("a", "c", "likes", {}),
            ("b", "d", "manages", {"edge_text": "   "}),
            ("b", "c", "owns"),
        ]
    )

    in_use = await adapter.get_edge_retrieval_texts_in_use(
        {"Alice works at Acme.", "Bob knows Carol.", "likes", "manages", "owns", "works_at", "x"}
    )

    assert in_use == {"Alice works at Acme.", "Bob knows Carol.", "likes", "manages", "owns"}
    assert adapter.full_reads == 1


@pytest.mark.asyncio
async def test_default_skips_the_read_when_nothing_is_asked():
    adapter = _FullReadAdapter([("a", "b", "likes", {})])

    assert await adapter.get_edge_retrieval_texts_in_use(set()) == set()
    assert await adapter.get_edge_retrieval_texts_in_use({""}) == set()
    assert adapter.full_reads == 0
