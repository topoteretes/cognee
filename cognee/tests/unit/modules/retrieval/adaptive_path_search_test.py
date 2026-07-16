import pytest

from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.path_search import AdaptivePathSearch, Path
from cognee.modules.retrieval.path_search import adaptive_path_search as aps_module


class MockScoredResult:
    """Mock class for vector search results."""

    def __init__(self, id, score, payload=None):
        self.id = id
        self.score = score
        self.payload = payload or {}


class StubVectorSearch:
    """Duck-typed stand-in for NodeEdgeVectorSearch with pre-baked results."""

    def __init__(self, node_results=None, edge_results=None):
        self.node_distances = {"Entity_name": node_results or []}
        self.edge_distances = edge_results or []
        self.query_list_length = None
        self.calls = []

    async def embed_and_retrieve_distances(self, **kwargs):
        self.calls.append(kwargs)

    def has_results(self):
        return any(bool(results) for results in self.node_distances.values()) or bool(
            self.edge_distances
        )

    def extract_relevant_node_ids(self):
        return [str(result.id) for results in self.node_distances.values() for result in results]


def make_graph(edge_specs):
    """Build a CogneeGraph from (source_id, target_id, relationship_type) tuples."""
    graph = CogneeGraph()
    for source_id, target_id, relationship_type in edge_specs:
        for node_id in (source_id, target_id):
            if graph.get_node(node_id) is None:
                graph.add_node(Node(node_id))
        edge = Edge(
            graph.get_node(source_id),
            graph.get_node(target_id),
            attributes={"relationship_type": relationship_type},
        )
        graph.add_edge(edge)
    return graph


def make_search(graph, node_scores, **overrides):
    """AdaptivePathSearch wired to an injected graph and stubbed vector results."""
    stub = StubVectorSearch(
        node_results=[MockScoredResult(node_id, score) for node_id, score in node_scores]
    )
    config = dict(
        num_seeds=3,
        walks_per_seed=5,
        max_depth=4,
        top_k=5,
        random_seed=42,
        memory_fragment=graph,
        vector_search=stub,
    )
    config.update(overrides)
    return AdaptivePathSearch(**config)


def path_keys(paths):
    return [path.dedup_key for path in paths]


@pytest.mark.parametrize(
    "field", ["num_seeds", "walks_per_seed", "max_depth", "top_k", "wide_search_top_k"]
)
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_config_raises(field, value):
    """Non-positive configuration values raise ValueError."""
    with pytest.raises(ValueError, match=field):
        AdaptivePathSearch(**{field: value})


@pytest.mark.asyncio
async def test_empty_query_raises():
    search = make_search(make_graph([("a", "b", "rel")]), [("a", 0.1)])
    with pytest.raises(ValueError, match="query"):
        await search.run("   ")


@pytest.mark.asyncio
async def test_no_vector_hits_returns_empty():
    """A vector search with no hits returns [] without touching the graph."""
    search = make_search(make_graph([("a", "b", "rel")]), node_scores=[])
    assert await search.run("anything") == []
    assert search.scored_candidates == []


@pytest.mark.asyncio
async def test_empty_graph_returns_empty():
    search = make_search(CogneeGraph(), [("a", 0.1)])
    assert await search.run("anything") == []


@pytest.mark.asyncio
async def test_injected_graph_is_reused(monkeypatch):
    """An injected memory fragment is used directly; no projection from the DB happens."""

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("get_memory_fragment must not be called with an injected graph")

    monkeypatch.setattr(aps_module, "get_memory_fragment", fail_if_called)

    graph = make_graph([("a", "b", "rel"), ("b", "c", "rel")])
    search = make_search(graph, [("a", 0.1)])
    results = await search.run("anything")

    assert results
    assert all(isinstance(path, Path) for path in results)


@pytest.mark.asyncio
async def test_fixed_seed_reproduces_paths():
    """The same random_seed yields identical paths across instances and repeated runs."""
    edge_specs = [
        ("hub", "a", "rel"),
        ("hub", "b", "rel"),
        ("hub", "c", "rel"),
        ("a", "d", "rel"),
        ("b", "e", "rel"),
        ("c", "f", "rel"),
        ("d", "e", "rel"),
    ]
    node_scores = [("hub", 0.1), ("a", 0.3), ("b", 0.5)]

    first_search = make_search(make_graph(edge_specs), node_scores)
    first = await first_search.run("anything")
    first_repeat = await first_search.run("anything")

    second_search = make_search(make_graph(edge_specs), node_scores)
    second = await second_search.run("anything")

    assert path_keys(first) == path_keys(first_repeat) == path_keys(second)
    assert [path.score for path in first] == [path.score for path in second]


@pytest.mark.asyncio
async def test_paths_never_revisit_nodes():
    """Walks on a cyclic graph never visit the same node twice."""
    graph = make_graph([("a", "b", "rel"), ("b", "c", "rel"), ("c", "a", "rel")])
    search = make_search(graph, [("a", 0.1), ("b", 0.2), ("c", 0.3)], max_depth=10)
    await search.run("anything")

    assert search.scored_candidates
    for path in search.scored_candidates:
        node_ids = [node.id for node in path.nodes]
        assert len(node_ids) == len(set(node_ids))


@pytest.mark.asyncio
async def test_dead_end_truncates_walk():
    """Walks hitting a dead end terminate cleanly instead of crashing or looping."""
    graph = make_graph([("a", "b", "rel")])
    search = make_search(graph, [("a", 0.1)], num_seeds=1, max_depth=10)
    results = await search.run("anything")

    assert len(results) == 1
    assert [node.id for node in results[0].nodes] == ["a", "b"]
    assert len(results[0].edges) == 1


@pytest.mark.asyncio
async def test_results_ordered_best_to_worst():
    """Returned paths are sorted by ascending score (lower distance = better)."""
    graph = make_graph(
        [
            ("hub", "close", "rel"),
            ("hub", "far", "rel"),
            ("close", "leaf1", "rel"),
            ("far", "leaf2", "rel"),
        ]
    )
    search = make_search(graph, [("hub", 0.1), ("close", 0.2), ("far", 1.9)], walks_per_seed=10)
    results = await search.run("anything")

    assert len(results) > 1
    scores = [path.score for path in results]
    assert scores == sorted(scores)
    assert search.scored_candidates == sorted(
        search.scored_candidates, key=lambda path: (path.score, path.dedup_key)
    )


@pytest.mark.asyncio
async def test_scored_candidates_retained_separately_from_selection():
    """All deduplicated scored paths are kept; run() returns only the top_k best."""
    graph = make_graph(
        [
            ("hub", "a", "rel"),
            ("hub", "b", "rel"),
            ("hub", "c", "rel"),
            ("a", "d", "rel"),
            ("b", "d", "rel"),
        ]
    )
    search = make_search(graph, [("hub", 0.1), ("a", 0.2), ("b", 0.3)], top_k=2, walks_per_seed=10)
    results = await search.run("anything")

    assert len(results) == 2
    assert len(search.scored_candidates) > 2
    assert results == search.scored_candidates[:2]
    assert len(set(path_keys(search.scored_candidates))) == len(search.scored_candidates)


@pytest.mark.asyncio
async def test_seeds_require_real_vector_hits():
    """Nodes without a vector hit (left at the penalty distance) are never seeds."""
    graph = make_graph([("a", "b", "rel"), ("c", "d", "rel")])
    search = make_search(graph, [("a", 0.1)], walks_per_seed=10)
    await search.run("anything")

    for path in search.scored_candidates:
        assert path.nodes[0].id == "a"
