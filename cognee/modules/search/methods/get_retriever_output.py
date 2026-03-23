from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    get_search_type_retriever_instance,
)
from cognee.modules.search.types import SearchType
from cognee.modules.retrieval.utils.access_tracking import update_node_access_timestamps
from cognee.shared.logging_utils import get_logger
from cognee.modules.observability import (
    new_span,
    COGNEE_SEARCH_TYPE,
    COGNEE_RESULT_COUNT,
    COGNEE_RESULT_SUMMARY,
)

logger = get_logger()


async def get_retriever_output(query_type: SearchType, query_text: str, **kwargs):
    graph_engine = await get_graph_engine()
    is_empty = await graph_engine.is_empty()

    if is_empty:
        logger.warning("Search attempt on an empty knowledge graph")

    # Recency-aware retrieval: over-fetch so reranking has headroom
    rsc = kwargs.get("retriever_specific_config") or {}
    recency_weight = rsc.get("recency_weight", 0.0)
    original_top_k = kwargs.get("top_k", 10)
    if recency_weight > 0:
        kwargs["top_k"] = original_top_k * 3

    retriever_instance = await get_search_type_retriever_instance(
        query_type=query_type, query_text=query_text, **kwargs
    )

    retriever_class = type(retriever_instance).__name__

    # Get raw result objects from retriever and forward to context and completion methods to avoid duplicate retrievals.
    with new_span("cognee.retrieval.get_objects") as span:
        span.set_attribute("cognee.retrieval.retriever", retriever_class)
        span.set_attribute(COGNEE_SEARCH_TYPE, query_type.value)
        retrieved_objects = await retriever_instance.get_retrieved_objects(query=query_text)
        obj_count = len(retrieved_objects) if isinstance(retrieved_objects, list) else 1
        span.set_attribute(COGNEE_RESULT_COUNT, obj_count)
        span.set_attribute(
            COGNEE_RESULT_SUMMARY,
            f"{retriever_class} retrieved {obj_count} object(s)",
        )

    # Recency reranking: blend distance scores with content freshness
    if recency_weight > 0 and isinstance(retrieved_objects, list) and len(retrieved_objects) > 1:
        retrieved_objects = _apply_recency_reranking(
            retrieved_objects, recency_weight, original_top_k
        )

    # Centralized access tracking for all retriever types
    if retrieved_objects:
        await update_node_access_timestamps(retrieved_objects)

    # Handle raw result object to extract context information
    with new_span("cognee.retrieval.get_context") as span:
        span.set_attribute("cognee.retrieval.retriever", retriever_class)
        context = await retriever_instance.get_context_from_objects(
            query=query_text, retrieved_objects=retrieved_objects
        )
        if isinstance(context, str):
            span.set_attribute("cognee.retrieval.context_length", len(context))
        elif isinstance(context, list):
            span.set_attribute("cognee.retrieval.context_items", len(context))

    completion = None
    if not kwargs.get(
        "only_context", False
    ):  # If only_context is True, skip getting completion. Performance optimization.
        # Handle raw result and context object to handle completion operation
        with new_span("cognee.retrieval.get_completion") as span:
            span.set_attribute("cognee.retrieval.retriever", retriever_class)
            completion = await retriever_instance.get_completion_from_context(
                query=query_text,
                retrieved_objects=retrieved_objects,
                context=context,
            )
            if isinstance(completion, str):
                span.set_attribute("cognee.retrieval.completion_length", len(completion))
            span.set_attribute(
                COGNEE_RESULT_SUMMARY,
                f"{retriever_class} generated completion",
            )

    search_result = SearchResultPayload(
        result_object=retrieved_objects,
        context=context,
        completion=completion,
        search_type=query_type,
        only_context=kwargs.get("only_context", False),
        dataset_name=kwargs.get("dataset").name if kwargs.get("dataset") else None,
        dataset_id=kwargs.get("dataset").id if kwargs.get("dataset") else None,
        dataset_tenant_id=kwargs.get("dataset").tenant_id if kwargs.get("dataset") else None,
    )

    return search_result


def _apply_recency_reranking(results: list, recency_weight: float, top_k: int) -> list:
    """Rerank ScoredResult objects by blending distance score with recency.

    Only affects ScoredResult objects whose payload contains ``updated_at``.
    Non-ScoredResult objects (e.g. Edge from graph retrievers) pass through unchanged.

    Args:
        results: Retrieved objects from a retriever.
        recency_weight: Blend factor (0.0 = pure relevance, 1.0 = pure recency).
        top_k: Final number of results to return.
    """
    from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult

    scored = [
        r
        for r in results
        if isinstance(r, ScoredResult) and r.payload and "updated_at" in r.payload
    ]
    non_scored = [r for r in results if r not in scored]

    if len(scored) < 2:
        return results[:top_k]

    timestamps = [r.payload["updated_at"] for r in scored]
    newest, oldest = max(timestamps), min(timestamps)
    time_range = newest - oldest

    if time_range == 0:
        # All items have the same timestamp — recency can't differentiate
        return results[:top_k]

    reranked = []
    for r in scored:
        age_penalty = (newest - r.payload["updated_at"]) / time_range  # 0=newest, 1=oldest
        blended = (1 - recency_weight) * r.score + recency_weight * age_penalty
        reranked.append((blended, r))

    reranked.sort(key=lambda x: x[0])
    return [r for _, r in reranked[:top_k]] + non_scored
