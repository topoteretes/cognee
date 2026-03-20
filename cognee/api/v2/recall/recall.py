from uuid import UUID
from typing import Optional


async def recall(
    query_text: str,
    query_type=None,
    datasets: Optional[list[str]] = None,
    dataset_ids: Optional[list[UUID]] = None,
    user=None,
    system_prompt: Optional[str] = None,
    system_prompt_path: Optional[str] = None,
    node_name: Optional[list[str]] = None,
    top_k: int = 10,
    verbose: bool = False,
    only_context: bool = False,
):
    """Search the knowledge graph for relevant information.

    This is a memory-oriented alias for ``cognee.search()``.  All arguments
    are forwarded unchanged.

    Args:
        query_text: Natural-language query.
        query_type: Search strategy (default ``SearchType.GRAPH_COMPLETION``).
        datasets: Dataset names to search within.
        dataset_ids: Dataset UUIDs to search within.
        user: User context for permissions.
        system_prompt: Custom system prompt text.
        system_prompt_path: Path to a system prompt file.
        node_name: Filter results to specific node sets.
        top_k: Maximum results to return (default *10*).
        verbose: Return verbose output.
        only_context: Return only the context that would be sent to the LLM.

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
        dataset_ids=dataset_ids,
        user=user,
        system_prompt=system_prompt,
        system_prompt_path=system_prompt_path,
        node_name=node_name,
        top_k=top_k,
        verbose=verbose,
        only_context=only_context,
    )
