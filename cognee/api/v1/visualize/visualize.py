from typing import Any, List, Literal, Optional, Tuple, Union
from uuid import UUID
from cognee.modules.users.models.User import User

from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
    aggregate_multi_user_graphs,
)
from cognee.modules.visualization.subgraph_data import (
    DEFAULT_MAX_NODES,
    DEFAULT_NEIGHBORHOOD_DEPTH,
    DEFAULT_SEED_TOP_K,
    fetch_visualization_graph_data,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.methods import get_default_user
from cognee.context_global_variables import set_database_global_context_variables
from cognee.shared.logging_utils import get_logger, setup_logging, ERROR


import asyncio


logger = get_logger()


async def visualize_graph(
    destination_file_path: str = None,
    include_session_events: bool = True,
    session_ids: list = None,
    user: Optional[User] = None,
    dataset: Optional[Union[str, UUID]] = "main_dataset",
    *,
    full: bool = False,
    scope: Optional[Literal["subgraph", "all"]] = None,
    query: Optional[str] = None,
    seed_node_ids: Optional[List[str]] = None,
    neighborhood_depth: int = DEFAULT_NEIGHBORHOOD_DEPTH,
    neighborhood_seed_top_k: int = DEFAULT_SEED_TOP_K,
    max_nodes: int = DEFAULT_MAX_NODES,
    recall_result: Optional[Any] = None,
) -> str:
    """Render the knowledge graph to a self-contained HTML file.

    By default renders a bounded subgraph around relevant seed nodes instead of
    the entire graph. Pass ``full=True`` or ``scope="all"`` for legacy full-graph
    rendering.

    Args:
        destination_file_path: Where to write the HTML (default: home dir).
        include_session_events: When True (default), best-effort collect the
            backend's search and feedback history from the session layer and
            show it on the Memory tab's timeline — searches as retrieval
            spotlights, rated answers as reinforcement (improve) events.
            Collection never fails the render; an unavailable session layer
            simply yields no events.
        session_ids: Restrict event collection to these sessions. Defaults to
            the user's most recently active sessions.
        user: User whose sessions are read. Defaults to the default user.
        dataset: Dataset to render, given by name or UUID. Wrapped into a
            single-element list for get_authorized_existing_datasets; the
            first authorized match selects which user+dataset database is
            visualized. Defaults to "main_dataset" (the same default used by
            add/cognify/remember). Pass None to skip dataset resolution and
            render the current context's graph.
        full: When True, render the entire graph (legacy behavior).
        scope: ``"all"`` is equivalent to ``full=True``; ``"subgraph"`` is the
            default bounded view.
        query: Optional query string; vector search hits become subgraph seeds.
        seed_node_ids: Explicit seed node IDs for neighborhood expansion.
        neighborhood_depth: k-hop expansion depth (default: 2).
        neighborhood_seed_top_k: Maximum number of seed nodes (default: 10).
        max_nodes: Hard cap on rendered nodes after expansion (default: 500).
        recall_result: Recall/search payload whose provenance seeds the subgraph.
    """
    if not user:
        user = await get_default_user()

    render_full_graph = full or scope == "all"

    # Only authorize when a dataset is given. get_authorized_existing_datasets
    # expects a list, so wrap the single dataset. With no dataset the context
    # is set with None: a no-op when access control is off, and an (expected)
    # error in multi-user mode where a dataset is required.
    if dataset:
        dataset = await get_authorized_existing_datasets([dataset], "read", user)

    async with set_database_global_context_variables(
        dataset[0].id if dataset else None,
        dataset[0].owner_id if dataset else None,
    ):
        graph_engine = await get_graph_engine()
        graph_data, subgraph_meta = await fetch_visualization_graph_data(
            graph_engine,
            full=render_full_graph,
            query=query,
            seed_node_ids=seed_node_ids,
            recall_result=recall_result,
            neighborhood_depth=neighborhood_depth,
            seed_top_k=neighborhood_seed_top_k,
            max_nodes=max_nodes,
            user=user,
            session_ids=session_ids,
        )
        if subgraph_meta.scope == "subgraph":
            logger.info(
                "Rendering bounded subgraph: seeds=%d source=%s depth=%d max_nodes=%d truncated=%s",
                len(subgraph_meta.seed_ids),
                subgraph_meta.seed_source,
                subgraph_meta.depth,
                subgraph_meta.max_nodes,
                subgraph_meta.truncated,
            )

        search_events = None
        if include_session_events:
            from cognee.modules.visualization.session_events import collect_session_events

            search_events = await collect_session_events(user=user, session_ids=session_ids)

        graph = await cognee_network_visualization(
            graph_data, destination_file_path, search_events=search_events
        )

        if destination_file_path:
            logger.info(f"The HTML file has been stored at path: {destination_file_path}")
        else:
            logger.info(
                "The HTML file has been stored on your home directory! Navigate there with cd ~"
            )

        return graph


async def visualize_search_subgraph(
    recall_responses: Any,
    destination_file_path: str = None,
    **kwargs: Any,
) -> str:
    """Visualize the subgraph behind a recall/search result."""
    return await visualize_graph(
        destination_file_path=destination_file_path,
        recall_result=recall_responses,
        **kwargs,
    )


async def visualize_multi_user_graph(
    user_dataset_pairs: List[Tuple[Any, Any]],
    destination_file_path: str = None,
) -> Any:
    """Generate a visualization combining graph data from multiple user+dataset pairs.

    Args:
        user_dataset_pairs: list of (User, Dataset) tuples to aggregate.
        destination_file_path: optional path to save the HTML output.

    Returns:
        The HTML visualization string.
    """
    graph_data = await aggregate_multi_user_graphs(user_dataset_pairs)

    graph = await cognee_network_visualization(graph_data, destination_file_path)

    if destination_file_path:
        logger.info(f"Multi-user visualization saved at: {destination_file_path}")
    else:
        logger.info(
            "Multi-user visualization saved to your home directory! Navigate there with cd ~"
        )

    return graph


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(visualize_graph())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
