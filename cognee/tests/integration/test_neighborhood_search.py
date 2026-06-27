"""Integration test: SearchType.NEIGHBORHOOD over a REAL Ladybug graph.

Validates ``get_neighborhood`` traversal plus the retriever's truncation and
serialization end-to-end on a real embedded Ladybug backend. Only the vector
seed-resolution step is stubbed (so the test needs no embeddings/LLM/network) —
known seed node IDs are fed directly while traversal runs for real.

Known graph (directed edges; traversal is undirected per the primitive):

    A -[KNOWS]-> B -[KNOWS]-> C      (A=hop0 seed; B=hop1; C=hop2)
    A -[WORKS_AT]-> D                (D=hop1 via a different edge type)
    E -[KNOWS]-> B                   (second component; E overlaps at B)

NOTE: edge-type filtering is NOT exercised against real data here. On the default
Ladybug/Kuzu backend, ``get_neighborhood``'s ``edge_types`` branch fails — Kuzu's
``ALL(rel IN r ...)`` rejects the recursive-rel binding of a variable-length path
(and referencing a query parameter inside that predicate crashes the parser). The
filter path is covered by the retriever unit test (mocked primitive) and is
expected to work only on the Neo4j/Postgres backends (extras-gated). The case is
kept below as a strict xfail so a future Ladybug fix flips it red.
"""

import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.modules.retrieval.neighborhood_retriever import NeighborhoodRetriever

_MODULE = "cognee.modules.retrieval.neighborhood_retriever"


class _FakeDataPoint:
    """Minimal DataPoint-like node stub (mirrors the adapter test convention)."""

    def __init__(self, id, name="", type="", **extra):
        self._data = {"id": str(id), "name": name, "type": type, **extra}

    def model_dump(self):
        return dict(self._data)


@pytest_asyncio.fixture
async def ladybug_graph():
    """Stand up a real Ladybug graph with the known topology, then tear it down."""
    tmp_dir = tempfile.mkdtemp(prefix="neighborhood_test_")
    adapter = LadybugAdapter(db_path=os.path.join(tmp_dir, "graph"))
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="A", name="Alice", type="Person"),
            _FakeDataPoint(id="B", name="Bob", type="Person"),
            _FakeDataPoint(id="C", name="Cara", type="Person"),
            _FakeDataPoint(id="D", name="Acme", type="Org"),
            _FakeDataPoint(id="E", name="Eve", type="Person"),
        ]
    )
    await adapter.add_edges(
        [
            ("A", "B", "KNOWS", {}),
            ("B", "C", "KNOWS", {}),
            ("A", "D", "WORKS_AT", {}),
            ("E", "B", "KNOWS", {}),
        ]
    )
    try:
        yield adapter
    finally:
        await adapter.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _seed_vector_search(*seed_scores):
    """Stub NodeEdgeVectorSearch so seed resolution yields the given (id, score)s."""
    vector_search = MagicMock()
    vector_search.embed_and_retrieve_distances = AsyncMock()
    vector_search.has_results = MagicMock(return_value=True)
    vector_search.node_distances = {
        "Entity_name": [SimpleNamespace(id=node_id, score=score) for node_id, score in seed_scores]
    }
    return vector_search


async def _run(adapter, retriever, vector_search, query="who is connected"):
    """Run the retriever end-to-end with a real graph and stubbed seed resolution."""
    unified = SimpleNamespace(graph=adapter, vector=object())
    with (
        patch(f"{_MODULE}.get_unified_engine", new_callable=AsyncMock) as mock_unified,
        patch(f"{_MODULE}.NodeEdgeVectorSearch", return_value=vector_search),
    ):
        mock_unified.return_value = unified
        completion = await retriever.get_completion(query)
    return completion[0]


@pytest.mark.asyncio
async def test_depth1_unfiltered_follows_all_edge_types(ladybug_graph):
    """seed=[A], depth=1, no filter -> {A, B, D}; C is out of depth."""
    retriever = NeighborhoodRetriever(depth=1)
    context = await _run(ladybug_graph, retriever, _seed_vector_search(("A", 0.1)))

    node_ids = {node["id"] for node in context["nodes"]}
    assert node_ids == {"A", "B", "D"}  # both KNOWS and WORKS_AT followed
    assert "C" not in node_ids
    assert context["truncated"] is False


@pytest.mark.asyncio
async def test_multi_seed_union_and_dedupe(ladybug_graph):
    """seeds=[A, E], depth=1 -> union of both ego-graphs; B appears exactly once."""
    retriever = NeighborhoodRetriever(depth=1)
    context = await _run(ladybug_graph, retriever, _seed_vector_search(("A", 0.1), ("E", 0.2)))

    node_ids = [node["id"] for node in context["nodes"]]
    assert set(node_ids) == {"A", "B", "D", "E"}  # A's {B,D} unioned with E's {B}
    assert len(node_ids) == len(set(node_ids))  # node dedupe: B not duplicated
    assert node_ids.count("B") == 1

    edge_keys = [
        (edge["source"], edge["target"], edge["relationship_name"]) for edge in context["edges"]
    ]
    assert len(edge_keys) == len(set(edge_keys))  # no duplicate edges in the union


@pytest.mark.asyncio
async def test_max_nodes_truncation_on_real_data(ladybug_graph):
    """seed=[A], depth=2, max_nodes=3 -> closest-first kept, farther nodes dropped."""
    retriever = NeighborhoodRetriever(depth=2, max_nodes=3)
    context = await _run(ladybug_graph, retriever, _seed_vector_search(("A", 0.1)))

    node_ids = {node["id"] for node in context["nodes"]}
    # Full depth-2 set is {A,B,C,D,E}; hops: A=0, B=1, D=1, C=2, E=2.
    # max_nodes=3 keeps the 3 closest: A, B, D.
    assert node_ids == {"A", "B", "D"}
    assert len(context["nodes"]) == 3
    assert "A" in node_ids  # seed always retained
    assert context["truncated"] is True


@pytest.mark.xfail(
    raises=RuntimeError,
    strict=True,
    reason=(
        "Ladybug/Kuzu get_neighborhood edge_types filter is broken: ALL(rel IN r ...) "
        "rejects a variable-length recursive-rel binding and a parameter inside the "
        "predicate crashes the parser. Works on Neo4j/Postgres (extras-gated)."
    ),
)
@pytest.mark.asyncio
async def test_depth2_edge_type_filter_xfail_on_ladybug(ladybug_graph):
    """seed=[A], depth=2, edge_types=['KNOWS'] -> would be {A,B,C} (D excluded).

    Expected to raise on Ladybug today; strict xfail flips red if Ladybug is fixed.
    """
    retriever = NeighborhoodRetriever(depth=2, edge_types=["KNOWS"])
    context = await _run(ladybug_graph, retriever, _seed_vector_search(("A", 0.1)))

    node_ids = {node["id"] for node in context["nodes"]}
    assert node_ids == {"A", "B", "C"}
    assert "D" not in node_ids
