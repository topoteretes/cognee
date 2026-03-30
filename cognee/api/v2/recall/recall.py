from uuid import UUID
from typing import Optional

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from typing_extensions import TypedDict

from cognee.modules.search.types import SearchType


class RecallKwargs(TypedDict, total=False):
    """Power-user overrides for recall(). Most users never need these."""

    dataset_ids: list[UUID]
    system_prompt: str
    system_prompt_path: str
    node_name: list[str]
    node_name_filter_operator: str
    only_context: bool
    session_id: str
    wide_search_top_k: int
    triplet_distance_penalty: float
    feedback_influence: float
    verbose: bool
    retriever_specific_config: dict
    user: object


async def recall(
    query_text: str,
    query_type: Optional[SearchType] = None,
    *,
    datasets: Optional[list[str]] = None,
    top_k: int = 10,
    auto_route: bool = True,
    **kwargs: Unpack[RecallKwargs],
) -> list:
    """Search the knowledge graph for relevant information.

    When ``query_type`` is omitted and ``auto_route`` is True (default),
    a lightweight rule-based classifier picks the best search strategy.
    Set ``auto_route=False`` to skip the classifier and use
    GRAPH_COMPLETION as the default, or pass ``query_type`` explicitly.

    Args:
        query_text: Natural-language query.
        query_type: Search strategy. When provided, the router is bypassed.
        datasets: Dataset names to search within.
        top_k: Maximum results to return (default *10*).
        auto_route: If True and query_type is None, classify the query
            automatically. If False, fall back to GRAPH_COMPLETION.
        **kwargs: Additional options -- see ``RecallKwargs``.

    Returns:
        Search results (same as ``cognee.search()``).
    """
    from cognee.api.v1.search import search

    routed_type = None

    if query_type is not None:
        # Explicit type: record override if the router would have picked differently
        if auto_route:
            from cognee.api.v2.recall.query_router import route_query, record_override

            result = route_query(query_text)
            routed_type = result.search_type
            record_override(routed_type, query_type)
    elif auto_route:
        from cognee.api.v2.recall.query_router import route_query

        result = route_query(query_text)
        query_type = result.search_type
    else:
        query_type = SearchType.GRAPH_COMPLETION

    return await search(
        query_text=query_text,
        query_type=query_type,
        datasets=datasets,
        top_k=top_k,
        **kwargs,
    )
