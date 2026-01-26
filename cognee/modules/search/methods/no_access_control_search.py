from typing import Any, List, Optional, Tuple, Type, Union

from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

from cognee.modules.search.methods.get_retriever_output import get_retriever_output

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
    wide_search_top_k: Optional[int] = 100,
    triplet_distance_penalty: Optional[float] = 3.5,
) -> Tuple[Any, Union[str, List[Edge]], List[Dataset]]:
    search_output = await get_retriever_output(
        query_type=query_type,
        query_text=query_text,
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
        top_k=top_k,
        node_type=node_type,
        node_name=node_name,
        save_interaction=save_interaction,
        last_k=last_k,
        only_context=only_context,
        session_id=session_id,
        wide_search_top_k=wide_search_top_k,
        triplet_distance_penalty=triplet_distance_penalty,
    )

    return search_output
