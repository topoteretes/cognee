from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Type, Union

from cognee.modules.observability import OtelStatusCode as StatusCode

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.graph.exceptions.exceptions import EntityNotFoundError
from cognee.modules.observability import (
    COGNEE_RESULT_SUMMARY,
    COGNEE_VECTOR_COLLECTION,
    COGNEE_VECTOR_RESULT_COUNT,
    new_span,
)
from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch
from cognee.modules.retrieval.utils.validate_queries import validate_queries
from cognee.shared.logging_utils import ERROR, get_logger
from cognee.exceptions import CogneeValidationError

if TYPE_CHECKING:
    from cognee.infrastructure.databases.unified import UnifiedStoreEngine

logger = get_logger(level=ERROR)

# Eval capture (SDK-529): cap on the candidate pool carried by one
# ``retrieval.candidates`` event. Batch mode searches every collection with no
# limit, so without a cap a single event could carry thousands of entries.
_CAPTURE_POOL_CAP = 500
_CAPTURE_STAGE = "brute_force_triplet_search"


def _distance_at(attributes: Dict, query_index: int, default: float) -> float:
    """The vector distance an element carries for one query; ``default`` when unset."""
    distances = attributes.get("vector_distance")
    if isinstance(distances, list) and query_index < len(distances):
        try:
            return float(distances[query_index])
        except (TypeError, ValueError):
            return default
    return default


def _scored_for_query(
    scored: Sequence, query_index: int, query_list_length: Optional[int]
) -> Sequence:
    """One query's slice of a ``NodeEdgeVectorSearch`` distance list.

    Single-query mode stores flat lists; batch mode stores one list per query.
    """
    if query_list_length is None:
        return scored or ()
    if query_index < len(scored):
        return scored[query_index] or ()
    return ()


def _retrieval_candidates_payload(
    vector_search: NodeEdgeVectorSearch,
    top_edges: Sequence[Edge],
    query_index: int,
    query_list_length: Optional[int],
    default_penalty: float,
) -> Dict:
    """Flat ``retrieval.candidates`` payload for one query (SDK-529). Scalars only.

    The pool is rebuilt from the ``id``/``score`` attributes of the vector-search
    results and the cut from the ids and attributes of the returned edges — never
    the ScoredResult / Node / Edge objects themselves (ScoredResult carries chunk
    text, Node/Edge back-reference the whole projected fragment).

    Pool: every scored candidate the distance-mapping step saw, as
    ``(id, collection, score)``, capped at ``_CAPTURE_POOL_CAP``. When the cap
    bites, each collection keeps the head of its (best-first) results up to an
    equal quota and the remaining room is filled in collection order, so one large
    collection cannot crowd the others out of the record.

    Cut: the returned triplets, ``score`` being the summed raw vector distance of
    source, edge, and target for this query — the un-weighted ranking input. The
    exact ranking key (importance-weight, feedback, and personal blends) is a
    closure inside ``CogneeGraph._calculate_query_top_triplet_importances`` and is
    not stored on the edge; it is deliberately not duplicated here.
    """
    per_collection: List[Tuple[str, Sequence]] = [
        (collection, _scored_for_query(scored, query_index, query_list_length))
        for collection, scored in vector_search.node_distances.items()
    ]
    per_collection.append(
        (
            vector_search.edge_collection,
            _scored_for_query(vector_search.edge_distances, query_index, query_list_length),
        )
    )

    pool_size = sum(len(scored) for _, scored in per_collection)
    if pool_size <= _CAPTURE_POOL_CAP:
        taken = [(collection, result) for collection, scored in per_collection for result in scored]
    else:
        quota = max(1, _CAPTURE_POOL_CAP // len(per_collection))
        taken = [
            (collection, result)
            for collection, scored in per_collection
            for result in scored[:quota]
        ]
        room = _CAPTURE_POOL_CAP - len(taken)
        for collection, scored in per_collection:
            if room <= 0:
                break
            extra = scored[quota : quota + room]
            taken.extend((collection, result) for result in extra)
            room -= len(extra)

    pool = [
        {
            "id": str(getattr(result, "id", "")),
            "collection": collection,
            "score": float(getattr(result, "score", default_penalty)),
        }
        for collection, result in taken
    ]

    cut = []
    for edge in top_edges:
        node1 = edge.node1
        node2 = edge.node2
        attributes = edge.attributes
        rel = (
            attributes.get("relationship_type")
            or attributes.get("relationship_name")
            or attributes.get("edge_text")
            or ""
        )
        score = (
            _distance_at(node1.attributes, query_index, default_penalty)
            + _distance_at(attributes, query_index, default_penalty)
            + _distance_at(node2.attributes, query_index, default_penalty)
        )
        cut.append(
            {"source": str(node1.id), "target": str(node2.id), "rel": str(rel), "score": score}
        )

    return {
        "query_index": query_index,
        "pool": pool,
        "top_k": cut,
        "pool_size": pool_size,
        "pool_truncated": pool_size > _CAPTURE_POOL_CAP,
        "cut_size": len(cut),
    }


def _emit_retrieval_candidates(
    vector_search: NodeEdgeVectorSearch,
    results: Union[List[Edge], List[List[Edge]]],
    query_list_length: Optional[int],
    triplet_distance_penalty: Optional[float],
) -> None:
    """Emit one ``retrieval.candidates`` event per query (SDK-529). Sync; never raises.

    Called only under ``is_active() and should_capture(...)`` — the OFF path never
    reaches this function, so it builds nothing when capture is disabled.
    """
    from cognee.modules.observability import capture as eval_capture

    try:
        default_penalty = (
            6.5 if triplet_distance_penalty is None else float(triplet_distance_penalty)
        )
        per_query = [(0, results)] if query_list_length is None else list(enumerate(results))
        for query_index, top_edges in per_query:
            eval_capture.emit(
                eval_capture.KIND_RETRIEVAL_CANDIDATES,
                _retrieval_candidates_payload(
                    vector_search, top_edges, query_index, query_list_length, default_penalty
                ),
                payload_kind="json",
                stage=_CAPTURE_STAGE,
            )
    except Exception as exc:
        # Capture never breaks the retrieval it observes.
        logger.debug("retrieval candidates capture skipped (%s)", exc)


def format_triplets(edges):
    """Formats edges into human-readable triplet strings."""
    triplets = []
    for edge in edges:
        node1 = edge.node1
        node2 = edge.node2
        edge_attributes = edge.attributes
        node1_attributes = node1.attributes
        node2_attributes = node2.attributes

        node1_info = {key: value for key, value in node1_attributes.items() if value is not None}
        node2_info = {key: value for key, value in node2_attributes.items() if value is not None}
        edge_info = {key: value for key, value in edge_attributes.items() if value is not None}

        triplet = f"Node1: {node1_info}\nEdge: {edge_info}\nNode2: {node2_info}\n\n\n"
        triplets.append(triplet)

    return "".join(triplets)


async def get_memory_fragment(
    properties_to_project: Optional[List[str]] = None,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
    node_name_filter_operator: str = "OR",
    relevant_ids_to_filter: Optional[List[str]] = None,
    memory_fragment_filter: Optional[List[dict]] = None,
    triplet_distance_penalty: Optional[float] = 6.5,
    feedback_influence: float = get_base_config().default_feedback_influence,
    graph_engine=None,
    neighborhood_depth: Optional[int] = None,
    neighborhood_seed_top_k: Optional[int] = 10,
) -> CogneeGraph:
    """Creates and initializes a CogneeGraph memory fragment with optional property projections."""
    if properties_to_project is None:
        properties_to_project = ["id", "description", "name", "type", "text", "importance_weight"]

    node_properties_to_project = list(properties_to_project)
    edge_properties_to_project = ["relationship_name", "edge_text", "edge_object_id"]

    if feedback_influence > 0.0:
        if "feedback_weight" not in node_properties_to_project:
            node_properties_to_project.append("feedback_weight")
        if "feedback_weight" not in edge_properties_to_project:
            edge_properties_to_project.append("feedback_weight")

    memory_fragment = CogneeGraph()

    try:
        if graph_engine is None:
            graph_engine = await get_graph_engine()

        if neighborhood_depth is not None and relevant_ids_to_filter:
            # Use neighborhood-based projection with seed nodes
            seed_ids = relevant_ids_to_filter[:neighborhood_seed_top_k]
            await memory_fragment.project_neighborhood_from_db(
                graph_engine,
                node_properties_to_project=node_properties_to_project,
                edge_properties_to_project=edge_properties_to_project,
                seed_node_ids=seed_ids,
                depth=neighborhood_depth,
                triplet_distance_penalty=triplet_distance_penalty,
                feedback_influence=feedback_influence,
            )
        elif neighborhood_depth is not None and not relevant_ids_to_filter:
            raise ValueError(
                "neighborhood_depth requires seed node IDs from vector search "
                "(set wide_search_top_k to a positive integer)"
            )
        else:
            await memory_fragment.project_graph_from_db(
                graph_engine,
                node_properties_to_project=node_properties_to_project,
                edge_properties_to_project=edge_properties_to_project,
                node_type=node_type,
                node_name=node_name,
                node_name_filter_operator=node_name_filter_operator,
                relevant_ids_to_filter=relevant_ids_to_filter,
                memory_fragment_filter=memory_fragment_filter or [],
                triplet_distance_penalty=triplet_distance_penalty,
                feedback_influence=feedback_influence,
            )
    except EntityNotFoundError:
        pass
    except Exception as e:
        logger.error(f"Error during memory fragment creation: {str(e)}")

    return memory_fragment


async def _get_top_triplet_importances(
    memory_fragment: Optional[CogneeGraph],
    vector_search: NodeEdgeVectorSearch,
    properties_to_project: Optional[List[str]],
    node_type: Optional[Type],
    node_name: Optional[List[str]],
    node_name_filter_operator: str,
    triplet_distance_penalty: float,
    feedback_influence: float,
    wide_search_limit: Optional[int],
    top_k: int,
    query_list_length: Optional[int] = None,
    graph_engine=None,
    neighborhood_depth: Optional[int] = None,
    neighborhood_seed_top_k: Optional[int] = 10,
    personal_weights: Optional[Dict[str, float]] = None,
) -> Union[List[Edge], List[List[Edge]]]:
    """Creates memory fragment (if needed), maps distances, and calculates top triplet importances.

    Args:
        query_list_length: Number of queries in batch mode (None for single-query mode).
            When None, node_distances/edge_distances are flat lists; when set, they are list-of-lists.
        graph_engine: Optional pre-created graph engine to pass through to
            ``get_memory_fragment()``.
        neighborhood_depth: If set, extract a k-hop neighborhood around seed nodes
            instead of projecting the full or ID-filtered graph.
        neighborhood_seed_top_k: Maximum number of seed nodes to use for neighborhood
            extraction. (default 10)
        personal_weights: Optional per-node ``prefers`` weights (node id -> weight in
            [0, 1]) applied to matching nodes so the triplet scorer can nudge ranking.

    Returns:
        List[Edge]: For single-query mode (query_list_length is None).
        List[List[Edge]]: For batch mode (query_list_length is set), one list per query.
    """
    if memory_fragment is None:
        if wide_search_limit is None:
            relevant_node_ids = None
        else:
            relevant_node_ids = vector_search.extract_relevant_node_ids()

        memory_fragment = await get_memory_fragment(
            properties_to_project=properties_to_project,
            node_type=node_type,
            node_name=node_name,
            node_name_filter_operator=node_name_filter_operator,
            relevant_ids_to_filter=relevant_node_ids,
            triplet_distance_penalty=triplet_distance_penalty,
            feedback_influence=feedback_influence,
            graph_engine=graph_engine,
            neighborhood_depth=neighborhood_depth,
            neighborhood_seed_top_k=neighborhood_seed_top_k,
        )

        # Re-score expansion nodes discovered via neighborhood traversal.
        # These nodes have no vector scores yet — run an ID-filtered vector
        # search so they participate in triplet ranking instead of getting
        # the default penalty score.
        if (
            neighborhood_depth is not None
            and relevant_node_ids
            and vector_search.query_vector is not None
        ):
            seed_set = set(relevant_node_ids)
            expansion_ids = [nid for nid in memory_fragment.nodes if nid not in seed_set]
            if expansion_ids:
                for collection_name in list(vector_search.node_distances.keys()):
                    try:
                        extra_scores = await vector_search.vector_engine.search(
                            collection_name=collection_name,
                            query_vector=vector_search.query_vector,
                            limit=len(expansion_ids),
                            node_name=expansion_ids,
                        )
                        if extra_scores:
                            if vector_search.query_list_length is None:
                                vector_search.node_distances[collection_name].extend(extra_scores)
                            else:
                                for qi, per_query in enumerate(extra_scores):
                                    if qi < len(vector_search.node_distances[collection_name]):
                                        vector_search.node_distances[collection_name][qi].extend(
                                            per_query
                                        )
                    except CollectionNotFoundError:
                        pass

    await memory_fragment.map_vector_distances_to_graph_nodes(
        node_distances=vector_search.node_distances, query_list_length=query_list_length
    )
    await memory_fragment.map_vector_distances_to_graph_edges(
        edge_distances=vector_search.edge_distances, query_list_length=query_list_length
    )

    # After projection and distance mapping so the weights land on the nodes
    # that actually made it into the fragment.
    if personal_weights:
        memory_fragment.apply_personal_weights(personal_weights)

    return await memory_fragment.calculate_top_triplet_importances(
        k=top_k,
        query_list_length=query_list_length,
        feedback_influence=feedback_influence,
    )


async def brute_force_triplet_search(
    query: Optional[str] = None,
    query_batch: Optional[List[str]] = None,
    top_k: int = 5,
    collections: Optional[List[str]] = None,
    properties_to_project: Optional[List[str]] = None,
    memory_fragment: Optional[CogneeGraph] = None,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
    node_name_filter_operator: str = "OR",
    wide_search_top_k: Optional[int] = 100,
    triplet_distance_penalty: Optional[float] = 6.5,
    feedback_influence: float = get_base_config().default_feedback_influence,
    unified_engine: Optional[UnifiedStoreEngine] = None,
    neighborhood_depth: Optional[int] = None,
    neighborhood_seed_top_k: Optional[int] = 10,
    personal_weights: Optional[Dict[str, float]] = None,
) -> Union[List[Edge], List[List[Edge]]]:
    """
    Performs a brute force search to retrieve the top triplets from the graph.

    Args:
        query (Optional[str]): The search query (single query mode). Exactly one of query or query_batch must be provided.
        query_batch (Optional[List[str]]): List of search queries (batch mode). Exactly one of query or query_batch must be provided.
        top_k (int): The number of top results to retrieve.
        collections (Optional[List[str]]): List of collections to query.
        properties_to_project (Optional[List[str]]): List of properties to project.
        memory_fragment (Optional[CogneeGraph]): Existing memory fragment to reuse.
        node_type: node type to filter
        node_name: node name to filter
        wide_search_top_k (Optional[int]): Number of initial elements to retrieve from collections.
            Ignored in batch mode (always None to project full graph).
        triplet_distance_penalty (Optional[float]): Default distance penalty in graph projection
        feedback_influence (float): Weight of feedback influence in range [0, 1]
        personal_weights (Optional[Dict[str, float]]): Per-node prefers weights
            (node id -> weight in [0, 1]) that nudge triplet ranking for the active user.

    Returns:
        List[Edge]: The top triplet results for single query mode (flat list).
        List[List[Edge]]: List of top triplet results (one per query) for batch mode (list-of-lists).

    Note:
        In single-query mode, node_distances and edge_distances are stored as flat lists.
        In batch mode, they are stored as list-of-lists (one list per query).
    """
    is_query_valid, msg = validate_queries(query, query_batch)
    if not is_query_valid:
        raise ValueError(msg)

    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if not 0.0 <= feedback_influence <= 1.0:
        raise CogneeValidationError(
            message="feedback_influence must be in range [0, 1]",
            name="InvalidFeedbackInfluenceError",
        )

    with new_span("cognee.retrieval.triplet_search") as otel_span:
        otel_span.set_attribute("cognee.retrieval.top_k", top_k)
        otel_span.set_attribute(
            "cognee.retrieval.mode", "batch" if query_batch is not None else "single"
        )

        query_list_length = len(query_batch) if query_batch is not None else None
        wide_search_limit = (
            None if query_list_length else (wide_search_top_k if node_name is None else None)
        )

        if collections is None:
            collections = [
                "Entity_name",
                "TextSummary_text",
                "EntityType_name",
                "DocumentChunk_text",
                # DLT rows live in their own collection — chunk search
                # is documents-only, but graph completion covers row text too.
                "DltRow_text",
            ]
        else:
            # Copy so the caller's list is never mutated. Callers such as
            # TripletSearchContextProvider pass a persistent ``self.collections``
            # that is shared across concurrent per-entity searches; appending the
            # edge collection in place would leak into that shared list.
            collections = list(collections)

        if "EdgeType_relationship_name" not in collections:
            collections.append("EdgeType_relationship_name")

        otel_span.set_attribute("cognee.retrieval.collection_count", len(collections))
        otel_span.set_attribute(COGNEE_VECTOR_COLLECTION, ", ".join(collections))

        # Eval capture (SDK-529). Imported lazily so ``import cognee`` never pays
        # for the capture package; is_active() is one global read once
        # initialized and is read once per search. The OFF path does nothing else:
        # no notes, no payload, no sampling roll.
        from cognee.modules.observability import capture as eval_capture

        capture_active = eval_capture.is_active()
        if capture_active:
            # The bounding settings of this search, mirrored from the span into
            # the run manifest so an offline eval can tie a candidate pool to the
            # knobs that produced it. No-ops outside a run scope.
            eval_capture.note("retrieval.top_k", top_k)
            eval_capture.note("retrieval.wide_search_top_k", wide_search_top_k)
            eval_capture.note("retrieval.neighborhood_seed_top_k", neighborhood_seed_top_k)
            eval_capture.note("retrieval.feedback_influence", feedback_influence)
            eval_capture.note("retrieval.mode", "batch" if query_batch is not None else "single")
            eval_capture.note("retrieval.collections", list(collections))

        try:
            vector_engine = unified_engine.vector if unified_engine else None
            graph_engine = unified_engine.graph if unified_engine else None

            vector_search = NodeEdgeVectorSearch(vector_engine=vector_engine)

            await vector_search.embed_and_retrieve_distances(
                query=None if query_list_length else query,
                query_batch=query_batch if query_list_length else None,
                collections=collections,
                wide_search_limit=wide_search_limit,
                node_name=node_name,
                node_name_filter_operator=node_name_filter_operator,
            )

            if query_batch is not None:
                otel_span.set_attribute("cognee.retrieval.batch_size", len(query_batch))

            if not vector_search.has_results():
                otel_span.set_attribute(COGNEE_VECTOR_RESULT_COUNT, 0)
                otel_span.set_attribute(COGNEE_RESULT_SUMMARY, "No vector results found")
                return [[] for _ in range(query_list_length)] if query_list_length else []

            results = await _get_top_triplet_importances(
                memory_fragment,
                vector_search,
                properties_to_project,
                node_type,
                node_name,
                node_name_filter_operator,
                triplet_distance_penalty,
                feedback_influence,
                wide_search_limit,
                top_k,
                query_list_length=query_list_length,
                graph_engine=graph_engine,
                neighborhood_depth=neighborhood_depth,
                neighborhood_seed_top_k=neighborhood_seed_top_k,
                personal_weights=personal_weights,
            )

            result_count = sum(len(r) for r in results) if query_list_length else len(results)
            otel_span.set_attribute(COGNEE_VECTOR_RESULT_COUNT, result_count)
            otel_span.set_attribute(
                COGNEE_RESULT_SUMMARY,
                f"Found {result_count} triplet(s) from {len(collections)} collection(s)",
            )

            # Candidate pool + final cut, after the result is computed and with no
            # await: emit() only buffers. Sampled per run for retrieval kinds; the
            # payload is built only when this search is sampled in.
            if capture_active and eval_capture.should_capture(
                eval_capture.KIND_RETRIEVAL_CANDIDATES
            ):
                _emit_retrieval_candidates(
                    vector_search, results, query_list_length, triplet_distance_penalty
                )

            return results
        except CollectionNotFoundError:
            return [[] for _ in range(query_list_length)] if query_list_length else []
        except Exception as error:
            otel_span.set_status(StatusCode.ERROR, str(error))
            otel_span.record_exception(error)

            logger.error(
                "Error during brute force search for query: %s. Error: %s",
                query_batch if query_list_length else [query],
                error,
            )
            raise error
