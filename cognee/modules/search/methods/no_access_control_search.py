from typing import Any, List, Optional, Tuple, Type, Union

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

from .get_search_type_tools import get_search_type_tools

logger = get_logger()


async def no_access_control_search(
    query_type: SearchType,
    query_text: str,
    system_prompt_path: str = "answer_simple_question.txt",
    system_prompt: Optional[str] = None,
    top_k: int = 10,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    save_interaction: bool = False,
    last_k: Optional[int] = None,
    only_context: bool = False,
    session_id: Optional[str] = None,
) -> Tuple[Any, Union[str, List[Edge]], List[Dataset]]:
    search_tools = await get_search_type_tools(
        query_type=query_type,
        query_text=query_text,
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
        top_k=top_k,
        node_type=node_type,
        node_name=node_name,
        save_interaction=save_interaction,
        last_k=last_k,
    )
    graph_engine = await get_graph_engine()
    is_empty = await graph_engine.is_empty()

    if is_empty:
        # TODO: we can log here, but not all search types use graph. Still keeping this here for reviewer input
        logger.warning("Search attempt on an empty knowledge graph")
    if len(search_tools) == 2:
        [get_completion, get_context] = search_tools

        if only_context:
            return None, await get_context(query_text), []

        context = await get_context(query_text)
        result = await get_completion(query_text, context, session_id=session_id)
    else:
        unknown_tool = search_tools[0]
        result = await unknown_tool(query_text)
        context = ""

    return result, context, []
