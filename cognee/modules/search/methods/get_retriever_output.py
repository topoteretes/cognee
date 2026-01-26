from uuid import UUID
from typing import Optional, Any, Dict, List, Union
from pydantic import BaseModel, ConfigDict, field_serializer
from pydantic.alias_generators import to_camel
from pydantic.main import IncEx

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.search.methods.get_search_type import get_search_type
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class SearchResultPayload(BaseModel):
    """Result payload from retriever classes."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    result_object: Any = None
    context: Optional[str] = None
    completion: Optional[Union[str, List[str]]] = None

    # TODO: Add return_type info
    search_type: SearchType
    only_context: bool = False

    dataset_name: Optional[str] = None
    dataset_id: Optional[UUID] = None
    dataset_tenant_id: Optional[UUID] = None

    @field_serializer("result_object")
    def serialize_complex_types(self, v: Any):
        # Handle serialization of complex types in result_object.
        # If result_object is a complex class, convert it to string here.
        if isinstance(v, list):
            return [str(i) if not isinstance(i, (int, float, dict, str)) else i for i in v]
        return v

    @property
    def result(self) -> Any:
        """Function used to determine search_result for users request.
        Return context if only_context is True, else return completion if it exists, else return result_object."""
        if self.only_context:
            return self.context
        elif self.completion:
            return self.completion
        else:
            return self.result_object


async def get_retriever_output(query_type: SearchType, query_text: str, **kwargs):
    graph_engine = await get_graph_engine()
    is_empty = await graph_engine.is_empty()

    if is_empty:
        logger.warning("Search attempt on an empty knowledge graph")

    retriever_instance = await get_search_type(
        query_type=query_type, query_text=query_text, **kwargs
    )

    # Get raw result objects from retriever and forward to context and completion methods to avoid duplicate retrievals.
    retrieved_objects = await retriever_instance.get_retrieved_objects(query=query_text)

    # Handle raw result object to extract context information
    context = await retriever_instance.get_context(retrieved_objects=retrieved_objects)

    completion = None
    if not kwargs.get(
        "only_context", False
    ):  # If only_context is True, skip getting completion. Performance optimization.
        # Handle raw result and context object to handle completion operation
        completion = await retriever_instance.get_completion(
            query=query_text,
            retrieved_objects=retrieved_objects,
            context=context,
            session_id=kwargs.get("session_id", None),
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
