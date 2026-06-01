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
    only_context = kwargs.get("only_context", False)
    persist_trace = kwargs.get("persist_trace", False)
    
    if not only_context:
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
    elif persist_trace:
        if hasattr(retriever_instance, "_use_session_cache") and retriever_instance._use_session_cache():
            from cognee.infrastructure.session.get_session_manager import get_session_manager
            from cognee.context_global_variables import session_user
            
            user = session_user.get()
            if user and getattr(user, "id", None):
                sm = get_session_manager()
                if sm and sm.is_available:
                    used_graph_element_ids = None
                    if hasattr(retriever_instance, "_extract_context_object_ids"):
                        used_graph_element_ids = retriever_instance._extract_context_object_ids(retrieved_objects)
                    
                    context_str = ""
                    if isinstance(context, str):
                        context_str = context
                    elif isinstance(context, list):
                        context_str = "\n\n".join([str(c) for c in context])
                    
                    await sm.add_qa(
                        user_id=str(user.id),
                        question=query_text,
                        context=context_str,
                        answer="",
                        session_id=getattr(retriever_instance, "session_id", None),
                        used_graph_element_ids=used_graph_element_ids,
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
