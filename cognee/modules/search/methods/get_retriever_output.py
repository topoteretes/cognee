from inspect import Parameter, signature

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.observability import (
    COGNEE_RESULT_COUNT,
    COGNEE_RESULT_SUMMARY,
    COGNEE_SEARCH_TYPE,
    new_span,
)
from cognee.modules.retrieval.utils.access_tracking import update_node_access_timestamps
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

    dataset = kwargs.get("dataset")
    allowed_node_ids = None
    if "user" in kwargs:
        from cognee.modules.data.methods import get_authorized_existing_datasets
        from cognee.modules.graph.methods.get_dataset_node_ids import get_dataset_node_ids

        if dataset is not None:
            allowed_dataset_ids = [dataset.id]
        else:
            search_datasets = await get_authorized_existing_datasets(
                datasets=None, permission_type="read", user=kwargs["user"]
            )
            allowed_dataset_ids = [ds.id for ds in search_datasets]

        allowed_node_ids = await get_dataset_node_ids(allowed_dataset_ids)
        retriever_instance.allowed_node_ids = allowed_node_ids

    retriever_class = type(retriever_instance).__name__
    only_context = kwargs.get("only_context", False)
    effective_query = query_text
    turn_preparation = None

    if not only_context:
        turn_preparation = await retriever_instance.prepare_session_turn_for_retrieval(query_text)
        if not turn_preparation.should_answer:
            return SearchResultPayload(
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
        effective_query = turn_preparation.effective_query or query_text

    # Get raw result objects from retriever and forward to context and completion methods to avoid duplicate retrievals.
    with new_span("cognee.retrieval.get_objects") as span:
        span.set_attribute("cognee.retrieval.retriever", retriever_class)
        span.set_attribute(COGNEE_SEARCH_TYPE, query_type.value)
        retrieved_objects = await retriever_instance.get_retrieved_objects(query=effective_query)

        # Enforce dataset isolation at the Python level
        if allowed_node_ids is not None:
            if isinstance(retrieved_objects, list):
                if retrieved_objects and isinstance(retrieved_objects[0], list):
                    # Batch mode (list-of-lists)
                    filtered_outer = []
                    for sublist in retrieved_objects:
                        filtered_inner = []
                        for item in sublist:
                            if hasattr(item, "node1"):
                                if (
                                    str(item.node1.id) in allowed_node_ids
                                    and str(item.node2.id) in allowed_node_ids
                                ):
                                    filtered_inner.append(item)
                            else:
                                item_id = str(
                                    getattr(item, "id", None)
                                    or (item.get("id") if isinstance(item, dict) else "")
                                )
                                if item_id in allowed_node_ids:
                                    filtered_inner.append(item)
                        filtered_outer.append(filtered_inner)
                    retrieved_objects = filtered_outer
                else:
                    # Single query mode (flat list)
                    filtered = []
                    for item in retrieved_objects:
                        if hasattr(item, "node1"):
                            if (
                                str(item.node1.id) in allowed_node_ids
                                and str(item.node2.id) in allowed_node_ids
                            ):
                                filtered.append(item)
                        else:
                            item_id = str(
                                getattr(item, "id", None)
                                or (item.get("id") if isinstance(item, dict) else "")
                            )
                            if item_id in allowed_node_ids:
                                filtered.append(item)
                    retrieved_objects = filtered
            elif isinstance(retrieved_objects, dict):
                # HybridRetriever output structure
                if "chunks" in retrieved_objects:
                    retrieved_objects["chunks"] = [
                        chunk
                        for chunk in retrieved_objects["chunks"]
                        if str(
                            getattr(chunk, "id", None)
                            or (chunk.get("id") if isinstance(chunk, dict) else "")
                        )
                        in allowed_node_ids
                    ]
                if "chunk_summaries" in retrieved_objects and isinstance(
                    retrieved_objects["chunk_summaries"], dict
                ):
                    retrieved_objects["chunk_summaries"] = {
                        chunk_id: summary
                        for chunk_id, summary in retrieved_objects["chunk_summaries"].items()
                        if str(chunk_id) in allowed_node_ids
                    }
                if "entities" in retrieved_objects:
                    retrieved_objects["entities"] = [
                        entity
                        for entity in retrieved_objects["entities"]
                        if str(
                            entity.get("id")
                            if isinstance(entity, dict)
                            else getattr(entity, "id", "")
                        )
                        in allowed_node_ids
                    ]
                if "facts" in retrieved_objects:
                    retrieved_objects["facts"] = [
                        fact
                        for fact in retrieved_objects["facts"]
                        if str(fact.get("source_node_id")) in allowed_node_ids
                        or str(fact.get("target_node_id")) in allowed_node_ids
                    ]

        obj_count = _count_retrieved_objects(retrieved_objects)
        span.set_attribute(COGNEE_RESULT_COUNT, obj_count)
        span.set_attribute(
            COGNEE_RESULT_SUMMARY,
            f"{retriever_class} retrieved {obj_count} object(s)",
        )

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
