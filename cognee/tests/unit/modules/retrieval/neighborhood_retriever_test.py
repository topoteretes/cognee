import inspect
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from cognee.modules.retrieval.neighborhood_retriever import NeighborhoodRetriever

_MODULE = "cognee.modules.retrieval.neighborhood_retriever"


def _scored(node_id, score):
    """A ScoredResult-like stub with the .id/.score attributes the retriever reads."""
    return SimpleNamespace(id=node_id, score=score)


def _make_vector_search(node_distances, has_results=True):
    """Build a mocked NodeEdgeVectorSearch instance.

    Mirrors the real single-query shape: node_distances is
    dict[collection -> list[scored]]; embed_and_retrieve_distances is awaited;
    has_results is a sync bool.
    """
    vector_search = MagicMock()
    vector_search.embed_and_retrieve_distances = AsyncMock()
    vector_search.has_results = MagicMock(return_value=has_results)
    vector_search.node_distances = node_distances
    return vector_search


def _make_unified_mock(graph_engine):
    unified = AsyncMock()
    unified.graph = graph_engine
    unified.vector = MagicMock()
    return unified


def _make_graph(is_empty=False, neighborhood=([], [])):
    graph = AsyncMock()
    graph.is_empty = AsyncMock(return_value=is_empty)
    graph.get_neighborhood = AsyncMock(return_value=neighborhood)
    return graph


def _patch_engine_and_vector(unified, vector_search):
    """Context managers patching get_unified_engine + NodeEdgeVectorSearch.

    Returns (unified_patch, vector_class_patch) so callers can enter both and
    inspect the vector class mock (e.g. call count).
    """
    unified_patch = patch(f"{_MODULE}.get_unified_engine", new_callable=AsyncMock)
    vector_patch = patch(f"{_MODULE}.NodeEdgeVectorSearch", return_value=vector_search)
    return unified_patch, vector_patch


# ---------------------------------------------------------------------------
# 1. Seed resolution: seed_top_k slicing + deterministic (score, id) ordering
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_seed_top_k_slicing_and_deterministic_order():
    """The 3 lowest-score ids are used as seeds, with ties broken by id."""
    # Seven candidates spread over two collections. "a" appears twice; the lower
    # score (0.1) must win. b/c/d all tie at 0.20 — only b and c make the top-3,
    # d is dropped purely by the id tie-break (b < c < d).
    node_distances = {
        "Entity_name": [
            _scored("a", 0.10),
            _scored("c", 0.20),
            _scored("b", 0.20),
            _scored("e", 0.30),
        ],
        "EntityType_name": [
            _scored("a", 0.90),  # duplicate id, worse score -> ignored
            _scored("d", 0.20),
            _scored("f", 0.40),
            _scored("g", 0.50),
        ],
    }
    vector_search = _make_vector_search(node_distances)
    graph = _make_graph(neighborhood=([("a", {})], []))
    unified_patch, vector_patch = _patch_engine_and_vector(_make_unified_mock(graph), vector_search)

    retriever = NeighborhoodRetriever(seed_top_k=3)
    with unified_patch as mock_unified, vector_patch:
        mock_unified.return_value = _make_unified_mock(graph)
        await retriever.get_retrieved_objects("query")

    graph.get_neighborhood.assert_awaited_once()
    seeds = graph.get_neighborhood.await_args.kwargs["node_ids"]
    assert seeds == ["a", "b", "c"]  # 0.10, then 0.20/0.20 tie-broken by id


# ---------------------------------------------------------------------------
# 2. BFS truncation correctness (closest-hops-first) + serialization shape
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bfs_truncation_drops_farthest_and_keeps_seed():
    """max_nodes cut keeps S, hop1, hop2 (closest first) and drops hop3."""
    # Chain S -A-> B -> C (hops 0,1,2,3).
    nodes = [
        ("S", {"name": "Seed", "type": "Entity", "extra": 1}),
        ("A", {"name": "Ay", "type": "Entity"}),
        ("B", {"name": "Bee", "type": "Entity"}),
        ("C", {"name": "Cee", "type": "Entity"}),
    ]
    edges = [
        ("S", "A", "KNOWS", {"w": 1}),
        ("A", "B", "KNOWS", {}),
        ("B", "C", "KNOWS", {}),
    ]
    retriever = NeighborhoodRetriever(depth=2, max_nodes=3)
    context = await retriever.get_context_from_objects(
        query="q", retrieved_objects={"seeds": ["S"], "nodes": nodes, "edges": edges}
    )

    kept_ids = [node["id"] for node in context["nodes"]]
    assert kept_ids == ["S", "A", "B"]  # closest-first; C (hop3) dropped
    assert "C" not in kept_ids
    assert context["truncated"] is True
    assert "S" in kept_ids

    # Edge to the dropped node is removed; only intra-kept edges survive.
    edge_pairs = {(edge["source"], edge["target"]) for edge in context["edges"]}
    assert edge_pairs == {("S", "A"), ("A", "B")}

    # Serialization shape: surfaced name/type, remaining props under "properties".
    seed_node = next(node for node in context["nodes"] if node["id"] == "S")
    assert seed_node == {
        "id": "S",
        "name": "Seed",
        "type": "Entity",
        "properties": {"extra": 1},
    }
    assert context["depth"] == 2


@pytest.mark.asyncio
async def test_no_truncation_when_max_nodes_high():
    """With max_nodes above the node count, everything is kept, truncated=False."""
    nodes = [
        ("S", {"name": "Seed", "type": "Entity"}),
        ("A", {"name": "Ay", "type": "Entity"}),
        ("B", {"name": "Bee", "type": "Entity"}),
        ("C", {"name": "Cee", "type": "Entity"}),
    ]
    edges = [
        ("S", "A", "KNOWS", {}),
        ("A", "B", "KNOWS", {}),
        ("B", "C", "KNOWS", {}),
    ]
    retriever = NeighborhoodRetriever(max_nodes=10)
    context = await retriever.get_context_from_objects(
        query="q", retrieved_objects={"seeds": ["S"], "nodes": nodes, "edges": edges}
    )

    assert {node["id"] for node in context["nodes"]} == {"S", "A", "B", "C"}
    assert context["truncated"] is False
    assert len(context["edges"]) == 3


# ---------------------------------------------------------------------------
# 3. max_nodes never drops seeds, even when seeds outnumber the cap
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_seeds_never_dropped_even_below_max_nodes():
    """3 seeds with max_nodes=1: all seeds retained; the non-seed neighbor drops."""
    nodes = [
        ("S1", {"name": "One", "type": "Entity"}),
        ("S2", {"name": "Two", "type": "Entity"}),
        ("S3", {"name": "Three", "type": "Entity"}),
        ("X", {"name": "Neighbor", "type": "Entity"}),
    ]
    edges = [("S1", "X", "KNOWS", {})]
    retriever = NeighborhoodRetriever(max_nodes=1)
    context = await retriever.get_context_from_objects(
        query="q",
        retrieved_objects={"seeds": ["S1", "S2", "S3"], "nodes": nodes, "edges": edges},
    )

    kept_ids = {node["id"] for node in context["nodes"]}
    assert {"S1", "S2", "S3"}.issubset(kept_ids)  # seeds always retained
    assert "X" not in kept_ids  # non-seed dropped under the cap
    assert context["truncated"] is True


# ---------------------------------------------------------------------------
# 4. edge_types pass-through to the primitive (verbatim)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_edge_types_passed_through_to_primitive():
    """edge_types reaches get_neighborhood unchanged, alongside depth/node_ids."""
    node_distances = {"Entity_name": [_scored("a", 0.1)]}
    vector_search = _make_vector_search(node_distances)
    graph = _make_graph(neighborhood=([("a", {})], []))
    unified_patch, vector_patch = _patch_engine_and_vector(_make_unified_mock(graph), vector_search)

    retriever = NeighborhoodRetriever(depth=2, edge_types=["KNOWS"])
    with unified_patch as mock_unified, vector_patch:
        mock_unified.return_value = _make_unified_mock(graph)
        await retriever.get_retrieved_objects("query")

    call = graph.get_neighborhood.await_args
    assert call.kwargs["edge_types"] == ["KNOWS"]
    assert call.kwargs["depth"] == 2
    assert call.kwargs["node_ids"] == ["a"]


# ---------------------------------------------------------------------------
# 5. No-LLM guarantee + exact return shape + vector engine touched exactly once
# ---------------------------------------------------------------------------
def test_module_imports_no_llm_machinery():
    """Static guarantee: the retriever module references no LLM entry points."""
    import cognee.modules.retrieval.neighborhood_retriever as module

    source = inspect.getsource(module)
    for forbidden in (
        "LLMGateway",
        "get_llm_client",
        "generate_completion",
        "acreate",
        "litellm",
        "llm_client",
    ):
        assert forbidden not in source, f"unexpected LLM reference: {forbidden}"


@pytest.mark.asyncio
async def test_completion_returns_structured_dict_and_vector_touched_once():
    """Full pipeline returns [context_dict] with exact keys; vector used once."""
    node_distances = {"Entity_name": [_scored("a", 0.1)]}
    vector_search = _make_vector_search(node_distances)
    nodes = [("a", {"name": "Ay", "type": "Entity"})]
    edges = []
    graph = _make_graph(neighborhood=(nodes, edges))
    unified_patch, vector_patch = _patch_engine_and_vector(_make_unified_mock(graph), vector_search)

    retriever = NeighborhoodRetriever()
    with unified_patch as mock_unified, vector_patch as mock_vector_class:
        mock_unified.return_value = _make_unified_mock(graph)
        completion = await retriever.get_completion("query")

    assert isinstance(completion, list) and len(completion) == 1
    context = completion[0]
    assert set(context.keys()) == {"seeds", "nodes", "edges", "truncated", "depth"}
    assert context["seeds"] == ["a"]

    # Vector engine touched exactly once: one NodeEdgeVectorSearch, one embed call.
    mock_vector_class.assert_called_once()
    vector_search.embed_and_retrieve_distances.assert_awaited_once()


# ---------------------------------------------------------------------------
# Empty cases
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_empty_graph_returns_empty_without_touching_vector_or_primitive():
    """is_empty() == True short-circuits before vector search or get_neighborhood."""
    vector_search = _make_vector_search({})
    graph = _make_graph(is_empty=True)
    unified_patch, vector_patch = _patch_engine_and_vector(_make_unified_mock(graph), vector_search)

    retriever = NeighborhoodRetriever()
    with unified_patch as mock_unified, vector_patch as mock_vector_class:
        mock_unified.return_value = _make_unified_mock(graph)
        retrieved = await retriever.get_retrieved_objects("query")
        context = await retriever.get_context_from_objects("query", retrieved)

    assert retrieved == {"seeds": [], "nodes": [], "edges": []}
    assert context == {"seeds": [], "nodes": [], "edges": [], "truncated": False, "depth": 2}
    mock_vector_class.assert_not_called()
    graph.get_neighborhood.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_vector_results_returns_empty_without_calling_primitive():
    """has_results() == False yields an empty subgraph and skips get_neighborhood."""
    vector_search = _make_vector_search({}, has_results=False)
    graph = _make_graph(is_empty=False)
    unified_patch, vector_patch = _patch_engine_and_vector(_make_unified_mock(graph), vector_search)

    retriever = NeighborhoodRetriever()
    with unified_patch as mock_unified, vector_patch:
        mock_unified.return_value = _make_unified_mock(graph)
        retrieved = await retriever.get_retrieved_objects("query")

    assert retrieved == {"seeds": [], "nodes": [], "edges": []}
    graph.get_neighborhood.assert_not_awaited()
