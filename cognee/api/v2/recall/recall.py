from uuid import UUID
from typing import Optional

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from typing_extensions import TypedDict


class RecallKwargs(TypedDict, total=False):
    """Power-user overrides for recall(). Most users never need these."""

    dataset_ids: list[UUID]
    system_prompt: str
    system_prompt_path: str
    node_name: list[str]
    only_context: bool
    session_id: str
    wide_search_top_k: int
    triplet_distance_penalty: float
    verbose: bool
    retriever_specific_config: dict
    user: object  # User context (resolved internally when None)


async def recall(
    query_text: str,
    query_type=None,
    *,
    datasets: Optional[list[str]] = None,
    top_k: int = 10,
    **kwargs: Unpack[RecallKwargs],
) -> list:
    """Search the knowledge graph for relevant information.

    This is a memory-oriented alias for ``cognee.search()``.  The most common
    parameters are explicit keyword arguments; power-user options can be passed
    via ``RecallKwargs`` (see class definition for available keys).

    Args:
        query_text: Natural-language query.
        query_type: Search strategy (default ``SearchType.GRAPH_COMPLETION``).
        datasets: Dataset names to search within.
        top_k: Maximum results to return (default *10*).
        **kwargs: Additional options — see ``RecallKwargs``.

    Returns:
        Search results (same as ``cognee.search()``).
    """
    from cognee.api.v1.search import search

    if query_type is None:
        from cognee.modules.search.types import SearchType

        query_type = SearchType.GRAPH_COMPLETION

    return await search(
        query_text=query_text,
        query_type=query_type,
        datasets=datasets,
        top_k=top_k,
        **kwargs,
    )
