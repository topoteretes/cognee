from uuid import UUID
from typing import Optional, Any, Dict, List, Union
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

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

    @property
    def result(self) -> Any:
        """Returns completion if available, otherwise context_object."""
        return self.completion if self.completion is not None else self.result_object


async def get_retriever_output(query_type: SearchType, query_text: str, **kwargs):
    graph_engine = await get_graph_engine()
    is_empty = await graph_engine.is_empty()

    if is_empty:
        # TODO: we can log here, but not all search types use graph. Still keeping this here for reviewer input
        logger.warning("Search attempt on an empty knowledge graph")

    retriever_instance = await get_search_type(
        query_type=query_type, query_text=query_text, **kwargs
    )

    retrieved_objects = await retriever_instance.get_retrieved_objects(query=query_text)

    context = await retriever_instance.get_context(retrieved_objects=retrieved_objects)

    completion = await retriever_instance.get_completion(
        query=query_text, retrieved_objects=retrieved_objects
    )

    search_result = SearchResultPayload(
        result_object=retrieved_objects,
        context=context,
        completion=completion,
        search_type=query_type,
    )

    return search_result
