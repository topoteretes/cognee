from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    get_search_type_retriever_instance,
)
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def get_retriever_output(query_type: SearchType, query_text: str, **kwargs):
    graph_engine = await get_graph_engine()
    is_empty = await graph_engine.is_empty()

    if is_empty:
        logger.warning("Search attempt on an empty knowledge graph")

    retriever_instance = await get_search_type_retriever_instance(
        query_type=query_type, query_text=query_text, **kwargs
    )

    # Get raw result objects from retriever and forward to context and completion methods to avoid duplicate retrievals.
    retrieved_objects = await retriever_instance.get_retrieved_objects(query=query_text)

    # Handle raw result object to extract context information
    context = await retriever_instance.get_context_from_objects(
        query=query_text, retrieved_objects=retrieved_objects
    )

    completion = None
    if not kwargs.get(
        "only_context", False
    ):  # If only_context is True, skip getting completion. Performance optimization.
        # Handle raw result and context object to handle completion operation
        completion = await retriever_instance.get_completion_from_context(
            query=query_text,
            retrieved_objects=retrieved_objects,
            context=context,
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
