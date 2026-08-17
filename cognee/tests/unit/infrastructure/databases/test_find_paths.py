"""Unit tests for the backend-agnostic GraphDBInterface.find_paths BFS primitive.

These exercise the default implementation directly with an in-memory fake adapter,
so they need no database, no LLM, and no API keys.
"""

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface


class _FakeGraph(GraphDBInterface):
    """Minimal graph that only implements get_connections.

    edges are (src, rel, dst). If ``directed_shape`` is True the connections are
    returned with the current node sometimes on the right of the triplet (the shape
    Neo4j uses for incoming edges), to test orientation handling. ``node_types`` maps a
    node id to its ``type`` so exclusion can be tested.
    """

    def __init__(self, edges, directed_shape=False, node_types=None):
        self.edges = edges
        self.directed_shape = directed_shape
        self.node_types = node_types or {}

    def _node(self, node_id):
        return {"id": node_id, "name": node_id, "type": self.node_types.get(node_id, "Entity")}

    async def get_connections(self, node_id):
        node_id = str(node_id)
        out = []
        for src, rel, dst in self.edges:
            triplet = (self._node(src), {"relationship_name": rel}, self._node(dst))
            if self.directed_shape:
                # Keep the real edge direction in the triplet regardless of which end
                # is the queried node, so incoming edges arrive as (neighbour, rel, current).
                if src == node_id or dst == node_id:
                    out.append(triplet)
            else:
                # Undirected shape (ladybug): current node is always the left side.
                if src == node_id:
                    out.append(triplet)
                elif dst == node_id:
                    out.append((self._node(dst), {"relationship_name": rel}, self._node(src)))
        return out


# Only get_connections is needed by find_paths; clear the remaining abstract methods so
# the fake can be instantiated for the test.
_FakeGraph.__abstractmethods__ = frozenset()

# A → B → C → D → E chain, plus an isolated node F.
CHAIN = [("A", "r1", "B"), ("B", "r2", "C"), ("C", "r3", "D"), ("D", "r4", "E")]


def _make(directed_shape=False):
    return _FakeGraph(CHAIN, directed_shape=directed_shape)


@pytest.mark.asyncio
async def test_find_paths_returns_shortest_chain():
    graph = _make()
    paths = await graph.find_paths("A", "E", max_depth=6)
    assert paths and paths[0]
    chain = [t[0]["id"] for t in paths[0]] + [paths[0][-1][2]["id"]]
    assert chain == ["A", "B", "C", "D", "E"]
    rels = [t[1]["relationship_name"] for t in paths[0]]
    assert rels == ["r1", "r2", "r3", "r4"]


@pytest.mark.asyncio
async def test_find_paths_no_path_when_unconnected():
    graph = _make()
    assert await graph.find_paths("A", "F", max_depth=6) == []


@pytest.mark.asyncio
async def test_find_paths_respects_max_depth():
    graph = _make()
    # A→E needs 4 hops; a 2-hop budget must not find it.
    assert await graph.find_paths("A", "E", max_depth=2) == []


@pytest.mark.asyncio
async def test_find_paths_same_source_and_target():
    graph = _make()
    assert await graph.find_paths("A", "A") == [[]]


@pytest.mark.asyncio
async def test_find_paths_handles_incoming_edge_orientation():
    # Neo4j-style: edges can arrive with the current node on the right of the triplet.
    graph = _make(directed_shape=True)
    paths = await graph.find_paths("A", "E", max_depth=6)
    assert paths and paths[0]
    chain = [t[0]["id"] for t in paths[0]] + [paths[0][-1][2]["id"]]
    assert chain == ["A", "B", "C", "D", "E"]


@pytest.mark.asyncio
async def test_find_paths_is_undirected_reachable_in_reverse():
    graph = _make(directed_shape=True)
    paths = await graph.find_paths("E", "A", max_depth=6)
    assert paths and paths[0]
    chain = [t[0]["id"] for t in paths[0]] + [paths[0][-1][2]["id"]]
    assert chain == ["E", "D", "C", "B", "A"]


# Mirrors the cognee graph: every entity hangs off a shared DocumentChunk via "contains".
# Without excluding the chunk, the shortest A→E path is the trivial 2-hop hop through it.
HUB_EDGES = [
    ("chunk", "contains", "A"),
    ("chunk", "contains", "B"),
    ("chunk", "contains", "C"),
    ("chunk", "contains", "D"),
    ("chunk", "contains", "E"),
    ("chunk", "contains", "Z"),  # unrelated entity, only reachable via the chunk
    ("A", "r1", "B"),
    ("B", "r2", "C"),
    ("C", "r3", "D"),
    ("D", "r4", "E"),
]


@pytest.mark.asyncio
async def test_find_paths_without_exclusion_takes_chunk_shortcut():
    # Demonstrates the bug: the shared chunk makes A→E look like a 2-hop path.
    graph = _FakeGraph(HUB_EDGES, node_types={"chunk": "DocumentChunk"})
    graph.__class__.__abstractmethods__ = frozenset()
    paths = await graph.find_paths("A", "E", max_depth=6)
    assert paths and len(paths[0]) == 2
    assert any(t[0]["id"] == "chunk" or t[2]["id"] == "chunk" for t in paths[0])


@pytest.mark.asyncio
async def test_find_paths_excluding_chunk_follows_real_chain():
    graph = _FakeGraph(HUB_EDGES, node_types={"chunk": "DocumentChunk"})
    graph.__class__.__abstractmethods__ = frozenset()
    paths = await graph.find_paths("A", "E", max_depth=6, excluded_node_types=["DocumentChunk"])
    assert paths and paths[0]
    chain = [t[0]["id"] for t in paths[0]] + [paths[0][-1][2]["id"]]
    assert chain == ["A", "B", "C", "D", "E"]


@pytest.mark.asyncio
async def test_find_paths_excluding_chunk_gives_no_path_for_unrelated_entity():
    # Z only connects through the chunk; excluding it must yield no path (the Maya case).
    graph = _FakeGraph(HUB_EDGES, node_types={"chunk": "DocumentChunk"})
    graph.__class__.__abstractmethods__ = frozenset()
    assert (
        await graph.find_paths("Z", "E", max_depth=6, excluded_node_types=["DocumentChunk"]) == []
    )
