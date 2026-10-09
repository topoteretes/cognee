"""Bounded subgraph selection for graph visualization.

``visualize_graph()`` renders a bounded subgraph by default instead of the whole
graph. Seeds are resolved by priority — explicit node ids, a recall/search
result's graph provenance, or a query string — and, when none of those are
given, the highest-degree nodes so a bare call still shows a representative
view. The seed neighborhood is read through the adapter's
``iter_bounded_neighborhood()``; adapters with a native implementation stop
at ``max_nodes`` in the store.
Pass ``full=True`` to render the entire graph (legacy behavior).
"""

from collections.abc import AsyncIterator
from typing import Any

from cognee.infrastructure.databases.graph.bounded_neighborhood import hop_distances
from cognee.infrastructure.databases.graph.graph_db_interface import EdgeData, Node
from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch
from cognee.modules.visualization.preprocessor import SEMANTIC_TYPE_KEY
from cognee.shared.logging_utils import get_logger

logger = get_logger("visualization.subgraph_data")

DEFAULT_NEIGHBORHOOD_DEPTH = 2
DEFAULT_SEED_TOP_K = 10
DEFAULT_MAX_NODES = 500
DEFAULT_WIDE_SEARCH_TOP_K = 100

# Node vector collections searched to turn a query into seed nodes — the same
# node-bearing collections search uses (keep in sync with the default list in
# retrieval/utils/brute_force_triplet_search.py). Edge collections are
# intentionally excluded: their hits are not graph nodes and cannot seed a
# neighborhood.
_SEED_VECTOR_COLLECTIONS = [
    "Entity_name",
    "TextSummary_text",
    "EntityType_name",
    "DocumentChunk_text",
]

# (nodes, edges) in the shape get_graph_data()/get_neighborhood() already return.
GraphData = tuple[list[Node], list[EdgeData]]


def _unique_preserve_order(node_ids: list[str]) -> list[str]:
    """Order-preserving de-duplication of stringified node ids."""
    return list(dict.fromkeys(str(node_id) for node_id in node_ids))


def _coerce_node_ids(value: Any) -> list[str]:
    """Node ids from a value that is either a ``node_ids`` list or a provenance
    mapping (``{"node_ids": [...]}``). Anything else yields ``[]``."""
    if isinstance(value, dict):
        value = value.get("node_ids")
    if isinstance(value, (list, tuple)):
        return [str(n) for n in value]
    return []


def resolve_seeds_from_recall(recall_result: Any) -> list[str]:
    """Seed node ids from a recall/search result's graph provenance.

    Handles the shapes cognee results actually carry node ids in: a mapping with
    a ``node_ids`` list (a citation/provenance payload), and recall()/remember()
    QA entries — as objects or dicts — that expose ``used_graph_element_ids``.
    Any other shape yields ``[]`` rather than raising.
    """
    if recall_result is None:
        return []

    if isinstance(recall_result, dict):
        seeds = _coerce_node_ids(recall_result.get("node_ids")) or _coerce_node_ids(
            recall_result.get("used_graph_element_ids")
        )
        return _unique_preserve_order(seeds)

    items = recall_result if isinstance(recall_result, (list, tuple)) else [recall_result]
    node_ids: list[str] = []
    for item in items:
        used = (
            item.get("used_graph_element_ids")
            if isinstance(item, dict)
            else getattr(item, "used_graph_element_ids", None)
        )
        node_ids.extend(_coerce_node_ids(used))
    return _unique_preserve_order(node_ids)


async def resolve_seeds_from_query(
    query: str,
    seed_top_k: int = DEFAULT_SEED_TOP_K,
    wide_search_top_k: int = DEFAULT_WIDE_SEARCH_TOP_K,
) -> list[str]:
    """Seed node ids for a query, ranked by vector distance (nearest first).

    ``NodeEdgeVectorSearch.extract_relevant_node_ids()`` returns an unordered
    set, so a plain slice of it is neither top-k nor deterministic. We rank the
    scored hits by distance here to get a genuinely most-relevant, stable top-k.
    """
    vector_search = NodeEdgeVectorSearch()
    await vector_search.embed_and_retrieve_distances(
        query=query,
        collections=list(_SEED_VECTOR_COLLECTIONS),
        wide_search_limit=wide_search_top_k,
    )
    if not vector_search.has_results():
        return []

    # Each collection's hits are ScoredResult(id, score); lower score = closer.
    scored = [
        (str(hit.id), hit.score)
        for hits in vector_search.node_distances.values()
        for hit in hits
        if getattr(hit, "id", None) is not None
    ]
    scored.sort(key=lambda pair: pair[1] if pair[1] is not None else float("inf"))
    return _unique_preserve_order([node_id for node_id, _ in scored])[:seed_top_k]


async def resolve_seeds_by_degree(graph_engine: Any, top_k: int) -> list[str]:
    """Highest-degree nodes as seeds — the default view when no seed is given.

    Delegates the ranking to the adapter, which lets a store that can
    aggregate do so. This used to read the whole graph via ``get_graph_data()``
    and count degree in Python: on a 5.6M-node / 35.6M-edge graph that is tens
    of gigabytes of Python objects built to keep ten ids, and the worker was
    OOM-killed before it could answer. Since this is the seed source for the
    *default* visualization — no query, no explicit seeds — that made the
    graph view unopenable at scale rather than merely slow.

    The in-memory count still exists as ``GraphDBInterface``'s inherited
    default, so an adapter without a native ranking keeps working.
    """
    method = getattr(graph_engine, "get_top_degree_node_ids", None)
    if callable(method):
        return await method(top_k)
    # Community registration permits duck-typed adapters, not only subclasses.
    from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

    return await GraphDBInterface.get_top_degree_node_ids(graph_engine, top_k)


async def resolve_seed_node_ids(
    graph_engine: Any,
    *,
    seed_node_ids: list[str] | None = None,
    recall_result: Any = None,
    query: str | None = None,
    seed_top_k: int = DEFAULT_SEED_TOP_K,
) -> tuple[list[str], str]:
    """Resolve seeds by priority: explicit ids > recall > query > degree."""
    if seed_node_ids:
        return _unique_preserve_order([str(n) for n in seed_node_ids])[:seed_top_k], "explicit"

    recall_seeds = resolve_seeds_from_recall(recall_result)
    if recall_seeds:
        return recall_seeds[:seed_top_k], "recall"

    if query:
        query_seeds = await resolve_seeds_from_query(query, seed_top_k=seed_top_k)
        if query_seeds:
            return query_seeds, "query"

    degree_seeds = await resolve_seeds_by_degree(graph_engine, seed_top_k)
    return (degree_seeds, "degree") if degree_seeds else ([], "none")


def truncate_subgraph(
    nodes_data: list[Node],
    edges_data: list[EdgeData],
    seed_ids: list[str],
    max_nodes: int,
) -> tuple[GraphData, bool]:
    """Cap the subgraph at ``max_nodes``, keeping seeds and their nearest nodes.

    Nodes are ranked by hop distance from the seeds (seeds first); edges are
    kept only when both endpoints survive, so no dangling edges remain.
    """
    if max_nodes <= 0 or len(nodes_data) <= max_nodes:
        return (nodes_data, edges_data), False

    hop_distance = hop_distances(edges_data, seed_ids)

    node_rank = {str(node_id): index for index, (node_id, _) in enumerate(nodes_data)}
    kept_nodes = sorted(
        nodes_data,
        key=lambda item: (hop_distance.get(str(item[0]), 10_000), node_rank[str(item[0])]),
    )[:max_nodes]
    kept_ids = {str(node_id) for node_id, _ in kept_nodes}
    kept_edges = [
        edge for edge in edges_data if str(edge[0]) in kept_ids and str(edge[1]) in kept_ids
    ]
    return (kept_nodes, kept_edges), True


ENTITY_NODE_TYPE = "Entity"


async def resolve_entity_types(graph_engine: Any, nodes_data: list[Node]) -> dict[str, str] | None:
    """The EntityType name of each Entity in ``nodes_data``, read from the store.

    A bounded read returns only the edges between its members, so an entity
    whose EntityType node was not admitted comes back with no way to tell its
    type. This asks the store for each entity's ``is_a`` target.

    Only Entity nodes are looked up: a type node as a seed would pull in every
    entity of that type. A failed lookup is logged and yields ``None``, so the
    read still returns, with those entities typed from the edges it has.
    """
    entity_ids = [
        str(node_id)
        for node_id, properties in nodes_data
        if properties.get("type") == ENTITY_NODE_TYPE
    ]
    if not entity_ids:
        return {}
    method = getattr(graph_engine, "get_entity_type_names", None)
    try:
        if callable(method):
            return await method(entity_ids)
        # Community registration permits duck-typed adapters, not only subclasses.
        from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

        return await GraphDBInterface.get_entity_type_names(graph_engine, entity_ids)
    except Exception:
        logger.warning(
            "Entity type lookup failed for %d entities; they keep the types the read itself shows.",
            len(entity_ids),
            exc_info=True,
        )
        return None


def with_entity_types(nodes_data: list[Node], entity_types: dict[str, str]) -> list[Node]:
    """``nodes_data`` with each resolved Entity's ``SEMANTIC_TYPE_KEY`` set."""
    return [
        (node_id, {**properties, SEMANTIC_TYPE_KEY: entity_types[str(node_id)]})
        if str(node_id) in entity_types
        else (node_id, properties)
        for node_id, properties in nodes_data
    ]


def _bounded_neighborhood(
    graph_engine: Any, seeds: list[str], depth: int, max_nodes: int, options: dict[str, Any]
) -> AsyncIterator[GraphData]:
    method = getattr(graph_engine, "iter_bounded_neighborhood", None)
    if callable(method):
        return method(seeds, depth, max_nodes, **options)
    # Community registration permits duck-typed adapters, not only subclasses.
    from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

    return GraphDBInterface.iter_bounded_neighborhood(
        graph_engine, seeds, depth, max_nodes, **options
    )


async def iter_seed_neighborhood(
    graph_engine: Any,
    seeds: list[str],
    depth: int,
    max_nodes: int,
    *,
    chunk_size: int,
    property_keys: list[str] | tuple[str, ...] | None = None,
) -> AsyncIterator[GraphData]:
    """The adapter's ``iter_bounded_neighborhood`` chunks for these seeds.

    A store with a native implementation stops at ``max_nodes`` instead of
    returning the whole neighbourhood for Python to cut. Every Entity in a
    chunk carries its semantic type as ``entity_type``, whether or not its
    EntityType node is in the read (see ``resolve_entity_types``).
    """
    # Only passed when asked for, so the unprojected read calls the adapter
    # exactly as it did before property projection had a caller.
    options: dict[str, Any] = {"chunk_size": chunk_size}
    if property_keys is not None:
        options["property_keys"] = list(property_keys)
    # A lookup that failed once is not retried for the rest of the read: a
    # store that cannot answer it would fail, and log, once per chunk.
    lookup_failed = False
    async for nodes_data, edges_data in _bounded_neighborhood(
        graph_engine, seeds, depth, max_nodes, options
    ):
        entity_types = None
        if not lookup_failed:
            entity_types = await resolve_entity_types(graph_engine, nodes_data)
            lookup_failed = entity_types is None
        yield with_entity_types(nodes_data, entity_types or {}), edges_data


async def expand_seed_neighborhood(
    graph_engine: Any, seeds: list[str], depth: int, max_nodes: int
) -> GraphData:
    """The seeds' neighbourhood capped at ``max_nodes``, as one ``(nodes, edges)``.

    The chunks are collected into one result; one chunk of ``max_nodes`` keeps
    it to a single node read.
    """
    nodes_data: list[Node] = []
    edges_data: list[EdgeData] = []
    async for chunk_nodes, chunk_edges in iter_seed_neighborhood(
        graph_engine, seeds, depth, max_nodes, chunk_size=max_nodes
    ):
        nodes_data.extend(chunk_nodes)
        edges_data.extend(chunk_edges)
    return nodes_data, edges_data


async def fetch_visualization_graph_data(
    graph_engine: Any,
    *,
    full: bool = False,
    query: str | None = None,
    seed_node_ids: list[str] | None = None,
    recall_result: Any = None,
    neighborhood_depth: int = DEFAULT_NEIGHBORHOOD_DEPTH,
    seed_top_k: int = DEFAULT_SEED_TOP_K,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> GraphData:
    """Return graph data for visualization: a bounded subgraph by default, or
    the whole graph when ``full`` is True. Output is the ``(nodes, edges)``
    format the renderer already consumes."""
    if neighborhood_depth < 1:
        raise ValueError("neighborhood_depth must be >= 1")
    if seed_top_k < 1:
        raise ValueError("seed_top_k must be >= 1")
    if max_nodes < 1:
        raise ValueError("max_nodes must be >= 1")

    if full:
        return await graph_engine.get_graph_data()

    seeds, source = await resolve_seed_node_ids(
        graph_engine,
        seed_node_ids=seed_node_ids,
        recall_result=recall_result,
        query=query,
        seed_top_k=seed_top_k,
    )
    if not seeds:
        logger.info("Subgraph visualization: no seeds resolved; rendering empty graph.")
        return ([], [])

    nodes_data, edges_data = await expand_seed_neighborhood(
        graph_engine, seeds, neighborhood_depth, max_nodes
    )
    logger.info(
        "Subgraph visualization: seeds=%d source=%s depth=%d max_nodes=%d nodes=%d edges=%d",
        len(seeds),
        source,
        neighborhood_depth,
        max_nodes,
        len(nodes_data),
        len(edges_data),
    )
    return (nodes_data, edges_data)
