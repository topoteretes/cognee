"""Personalized-PageRank triplet search.

A graph-structural alternative to ``brute_force_triplet_search``. The existing
brute-force scorer ranks every triplet purely by the vector distance of its own
elements, so the graph decides *which* facts are candidates but not *how* they
rank. This search adds the missing structural signal: it seeds Personalized
PageRank on the vector hits and lets relevance spread through the graph, so a
"bridge" fact that connects several query-relevant entities can surface even when
its own text is not similar to the query. This is the mechanism HippoRAG
(Gutierrez et al., NeurIPS 2024) uses for single-step multi-hop retrieval, applied
to the entity graph Cognee already builds.

The vector step, graph projection, and distance mapping are reused verbatim from
the brute-force path; only the final ranking differs.
"""

from __future__ import annotations

import heapq
from typing import List, Optional, Type

import networkx as nx

from cognee.base_config import get_base_config
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.utils.brute_force_triplet_search import get_memory_fragment
from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch
from cognee.shared.logging_utils import ERROR, get_logger

logger = get_logger(level=ERROR)

# networkx damping default; how far activation spreads from the seeds. Higher
# spreads wider, lower stays closer to the query's direct hits. Tunable.
DEFAULT_PPR_ALPHA = 0.85
# Blend between structural PageRank (1.0) and direct vector similarity (0.0) in
# the final ranking. 0.5 gives both an equal say.
DEFAULT_PPR_WEIGHT = 0.5
# One-hop expansion around the seeds by default, so bridge nodes that are not
# themselves vector hits are present in the fragment and can accrue activation.
DEFAULT_NEIGHBORHOOD_DEPTH = 1

DEFAULT_COLLECTIONS = [
    "Entity_name",
    "TextSummary_text",
    "EntityType_name",
    "DocumentChunk_text",
]
EDGE_COLLECTION = "EdgeType_relationship_name"


def _distance_of(element, penalty: float) -> float:
    """Read a graph element's (single-query) vector distance, defaulting to the
    penalty when it has no real match."""
    distances = element.attributes.get("vector_distance")
    if isinstance(distances, list) and distances:
        try:
            return float(distances[0])
        except (TypeError, ValueError):
            return penalty
    return penalty


def _similarity_of(element, penalty: float) -> float:
    """Convert a cosine distance in [0, 2] to a similarity in [0, 1]. Anything at
    or above the penalty (a non-match) maps to 0, so it never contributes mass."""
    distance = _distance_of(element, penalty)
    if distance >= penalty:
        return 0.0
    similarity = 1.0 - (distance / 2.0)
    return similarity if similarity > 0.0 else 0.0


def _build_nx_graph(memory_fragment: CogneeGraph) -> nx.Graph:
    """Build an undirected graph from the projected fragment. Undirected so
    relevance flows both ways along a relationship; parallel relationships between
    the same pair collapse, which is fine because PageRank scores nodes."""
    graph = nx.Graph()
    graph.add_nodes_from(memory_fragment.nodes.keys())
    for edge in memory_fragment.edges:
        graph.add_edge(edge.node1.id, edge.node2.id)
    return graph


def _personalized_pagerank(graph: nx.Graph, personalization: dict, alpha: float) -> dict:
    """Run Personalized PageRank, falling back to an empty result (which degrades
    the ranking to similarity-only) rather than raising, so a numerical or
    connectivity edge case can never break retrieval."""
    if graph.number_of_nodes() == 0 or not personalization:
        return {}
    try:
        return nx.pagerank(graph, alpha=alpha, personalization=personalization)
    except Exception as exc:  # noqa: BLE001 - PageRank must never break retrieval
        logger.warning(
            "Personalized PageRank unavailable, falling back to similarity-only ranking: %s",
            exc,
        )
        return {}


async def ppr_triplet_search(
    query: str,
    top_k: int = 5,
    collections: Optional[List[str]] = None,
    properties_to_project: Optional[List[str]] = None,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
    node_name_filter_operator: str = "OR",
    wide_search_top_k: Optional[int] = 100,
    triplet_distance_penalty: Optional[float] = 6.5,
    feedback_influence: float = None,
    ppr_alpha: float = DEFAULT_PPR_ALPHA,
    ppr_weight: float = DEFAULT_PPR_WEIGHT,
    neighborhood_depth: Optional[int] = DEFAULT_NEIGHBORHOOD_DEPTH,
    neighborhood_seed_top_k: int = 10,
    unified_engine=None,
) -> List[Edge]:
    """Retrieve the top_k triplets for a single query, ranked by a blend of
    Personalized PageRank (graph structure) and direct vector similarity.

    Returns a flat list of ``Edge`` triplets (same contract as the single-query
    path of ``brute_force_triplet_search``). Returns ``[]`` when the graph or the
    vector search yields nothing.
    """
    if not query or not isinstance(query, str) or not query.strip():
        raise ValueError("The query must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if feedback_influence is None:
        feedback_influence = get_base_config().default_feedback_influence
    if not 0.0 <= feedback_influence <= 1.0:
        raise CogneeValidationError(
            message="feedback_influence must be in range [0, 1]",
            name="InvalidFeedbackInfluenceError",
        )
    ppr_weight = max(0.0, min(1.0, ppr_weight))

    penalty = triplet_distance_penalty if triplet_distance_penalty is not None else 6.5

    collections = list(collections) if collections else list(DEFAULT_COLLECTIONS)
    if EDGE_COLLECTION not in collections:
        collections.append(EDGE_COLLECTION)

    # A filtered (node_name) search projects the whole nodeset rather than an
    # ID-filtered seed set, so there is no wide-search limit and no seed IDs.
    wide_search_limit = wide_search_top_k if node_name is None else None

    try:
        vector_engine = unified_engine.vector if unified_engine else None
        graph_engine = unified_engine.graph if unified_engine else None

        vector_search = NodeEdgeVectorSearch(vector_engine=vector_engine)
        await vector_search.embed_and_retrieve_distances(
            query=query,
            collections=collections,
            wide_search_limit=wide_search_limit,
            node_name=node_name,
            node_name_filter_operator=node_name_filter_operator,
        )
        if not vector_search.has_results():
            return []

        seed_ids = (
            vector_search.extract_relevant_node_ids() if wide_search_limit is not None else None
        )

        # Neighborhood expansion needs seed IDs; skip it on the filtered path.
        effective_depth = neighborhood_depth if seed_ids else None

        memory_fragment = await get_memory_fragment(
            properties_to_project=properties_to_project,
            node_type=node_type,
            node_name=node_name,
            node_name_filter_operator=node_name_filter_operator,
            relevant_ids_to_filter=seed_ids,
            triplet_distance_penalty=penalty,
            feedback_influence=feedback_influence,
            graph_engine=graph_engine,
            neighborhood_depth=effective_depth,
            neighborhood_seed_top_k=neighborhood_seed_top_k,
        )

        # If neighborhood expansion produced nothing (e.g. an adapter without
        # get_neighborhood support), retry with a plain ID-filtered projection so
        # PPR still has a fragment to rank.
        if not memory_fragment.edges and effective_depth is not None:
            memory_fragment = await get_memory_fragment(
                properties_to_project=properties_to_project,
                node_type=node_type,
                node_name=node_name,
                node_name_filter_operator=node_name_filter_operator,
                relevant_ids_to_filter=seed_ids,
                triplet_distance_penalty=penalty,
                feedback_influence=feedback_influence,
                graph_engine=graph_engine,
                neighborhood_depth=None,
            )

        await memory_fragment.map_vector_distances_to_graph_nodes(
            node_distances=vector_search.node_distances, query_list_length=None
        )
        await memory_fragment.map_vector_distances_to_graph_edges(
            edge_distances=vector_search.edge_distances, query_list_length=None
        )

        if not memory_fragment.edges:
            return []

        # Restart mass = vector similarity: the model attends most to nodes the
        # query directly matched, then PageRank spreads that mass through edges.
        personalization = {}
        for node_id, node in memory_fragment.nodes.items():
            similarity = _similarity_of(node, penalty)
            if similarity > 0.0:
                personalization[node_id] = similarity

        graph = _build_nx_graph(memory_fragment)
        ppr_scores = _personalized_pagerank(graph, personalization, ppr_alpha)
        max_ppr = max(ppr_scores.values()) if ppr_scores else 0.0

        def node_relevance(node: Node) -> float:
            similarity = _similarity_of(node, penalty)
            ppr_norm = (ppr_scores.get(node.id, 0.0) / max_ppr) if max_ppr > 0.0 else 0.0
            return ppr_weight * ppr_norm + (1.0 - ppr_weight) * similarity

        def edge_score(edge: Edge) -> float:
            # Higher is better: both endpoints' blended relevance plus the edge's
            # own direct similarity. No penalty term can dominate because
            # non-matches contribute 0, not a large distance.
            return (
                node_relevance(edge.node1)
                + node_relevance(edge.node2)
                + _similarity_of(edge, penalty)
            )

        return heapq.nlargest(top_k, memory_fragment.edges, key=edge_score)

    except CollectionNotFoundError:
        return []
