from typing import Any, List, Tuple

from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
    aggregate_multi_user_graphs,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import get_logger, setup_logging, ERROR


import asyncio


logger = get_logger()


async def visualize_graph(
    destination_file_path: str = None,
    include_session_events: bool = True,
    session_ids: list = None,
    user=None,
) -> str:
    """Render the knowledge graph to a self-contained HTML file.

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
    """
    graph_engine = await get_graph_engine()
    graph_data = await graph_engine.get_graph_data()

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
