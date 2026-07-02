from inspect import Parameter, signature
from typing import Any

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.observability import (
    COGNEE_RESULT_COUNT,
    COGNEE_RESULT_SUMMARY,
    COGNEE_SEARCH_TYPE,
    new_span,
)
from cognee.modules.retrieval.utils.access_tracking import update_node_access_timestamps
from cognee.modules.retrieval.utils.used_graph_elements import (
    build_retrieved_subgraph,
    supports_subgraph_search_type,
)
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    get_search_type_retriever_instance,
)
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def _method_accepts_kwarg(method, name: str) -> bool:
    parameters = signature(method).parameters.values()
    return any(
        parameter.kind == Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in parameters
    )


def _subgraph_counts(subgraph: Any) -> tuple[int, int]:
    if subgraph is None:
        return 0, 0
    if isinstance(subgraph, list):
        node_count = sum(len(item.get("nodes", [])) for item in subgraph if isinstance(item, dict))
        edge_count = sum(len(item.get("edges", [])) for item in subgraph if isinstance(item, dict))
        return node_count, edge_count
    if isinstance(subgraph, dict):
        return len(subgraph.get("nodes", [])), len(subgraph.get("edges", []))
    return 0, 0


async def get_retriever_output(
    query_type: SearchType, query_text: str, **kwargs
) -> SearchResultPayload:
    graph_engine = await get_graph_engine()
    is_empty = await graph_engine.is_empty()

    if is_empty:
        logger.warning("Search attempt on an empty knowledge graph")

    retriever_instance = await get_search_type_retriever_instance(
        query_type=query_type, query_text=query_text, **kwargs
    )

    retriever_class = type(retriever_instance).__name__
    only_context = kwargs.get("only_context", False)
    include_subgraph = kwargs.get("include_subgraph", False)
    effective_query = query_text
    turn_preparation = None

    if not only_context:
        turn_preparation = await retriever_instance.prepare_session_turn_for_retrieval(query_text)
        if not turn_preparation.should_answer:
            payload_kwargs = dict(
                result_object=None,
                context=None,
                completion=[turn_preparation.response_to_user or "Got it."],
                search_type=query_type,
                only_context=False,
                dataset_name=kwargs.get("dataset").name if kwargs.get("dataset") else None,
                dataset_id=kwargs.get("dataset").id if kwargs.get("dataset") else None,
                dataset_tenant_id=kwargs.get("dataset").tenant_id
                if kwargs.get("dataset")
                else None,
            )
            if include_subgraph:
                payload_kwargs["retrieved_subgraph"] = None
            return SearchResultPayload(**payload_kwargs)
        effective_query = turn_preparation.effective_query or query_text

    # Get raw result objects from retriever and forward to context and completion methods to avoid duplicate retrievals.
    retrieved_subgraph = None
    with new_span("cognee.retrieval.get_objects") as span:
        span.set_attribute("cognee.retrieval.retriever", retriever_class)
        span.set_attribute(COGNEE_SEARCH_TYPE, query_type.value)
        retrieved_objects = await retriever_instance.get_retrieved_objects(query=effective_query)
        obj_count = _count_retrieved_objects(retrieved_objects)
        span.set_attribute(COGNEE_RESULT_COUNT, obj_count)
        span.set_attribute(
            COGNEE_RESULT_SUMMARY,
            f"{retriever_class} retrieved {obj_count} object(s)",
        )
        if include_subgraph:
            retrieved_subgraph = (
                build_retrieved_subgraph(retrieved_objects, query_type)
                if supports_subgraph_search_type(query_type)
                else None
            )
            node_count, edge_count = _subgraph_counts(retrieved_subgraph)
            span.set_attribute("cognee.retrieval.subgraph_node_count", node_count)
            span.set_attribute("cognee.retrieval.subgraph_edge_count", edge_count)
        else:
            retrieved_subgraph = None

    # Centralized access tracking for all retriever types
    if retrieved_objects:
        await update_node_access_timestamps(retrieved_objects)

    # Handle raw result object to extract context information
    with new_span("cognee.retrieval.get_context") as span:
        span.set_attribute("cognee.retrieval.retriever", retriever_class)
        context = await retriever_instance.get_context_from_objects(
            query=effective_query, retrieved_objects=retrieved_objects
        )
        if isinstance(context, str):
            span.set_attribute("cognee.retrieval.context_length", len(context))
        elif isinstance(context, list):
            span.set_attribute("cognee.retrieval.context_items", len(context))

    completion = None
    if not only_context:  # If only_context is True, skip completion. Performance optimization.
        # Handle raw result and context object to handle completion operation
        with new_span("cognee.retrieval.get_completion") as span:
            span.set_attribute("cognee.retrieval.retriever", retriever_class)
            completion_kwargs = {
                "query": query_text,
                "retrieved_objects": retrieved_objects,
                "context": context,
            }
            completion_method = retriever_instance.get_completion_from_context
            if _method_accepts_kwarg(completion_method, "effective_query"):
                completion_kwargs["effective_query"] = effective_query
            if _method_accepts_kwarg(completion_method, "turn_preparation"):
                completion_kwargs["turn_preparation"] = turn_preparation
            completion = await completion_method(**completion_kwargs)
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
        only_context=only_context,
        dataset_name=kwargs.get("dataset").name if kwargs.get("dataset") else None,
        dataset_id=kwargs.get("dataset").id if kwargs.get("dataset") else None,
        dataset_tenant_id=kwargs.get("dataset").tenant_id if kwargs.get("dataset") else None,
        retrieved_subgraph=retrieved_subgraph if include_subgraph else None,
    )

    return search_result


def _count_retrieved_objects(retrieved_objects) -> int:
    if retrieved_objects is None:
        return 0
    if isinstance(retrieved_objects, list):
        return len(retrieved_objects)
    if isinstance(retrieved_objects, dict):
        list_counts = [
            len(value) for value in retrieved_objects.values() if isinstance(value, list)
        ]
        if list_counts:
            return sum(list_counts)
        return 1
    return 1
