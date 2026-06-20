from typing import Any, List, Optional, Tuple, Union
from uuid import UUID

from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
    aggregate_multi_user_graphs,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.context_global_variables import (
    backend_access_control_enabled,
    set_database_global_context_variables,
)
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger, setup_logging, ERROR


import asyncio


logger = get_logger()


async def visualize_graph(
    destination_file_path: str = None,
    dataset: Optional[Union[str, UUID]] = None,
    user: Optional[User] = None,
) -> str:
    if backend_access_control_enabled():
        if user is None:
            user = await get_default_user()

        # get_authorized_existing_datasets expects a list, so wrap the single dataset.
        authorized_datasets = await get_authorized_existing_datasets([dataset], "read", user)

        if authorized_datasets:
            dataset = authorized_datasets[0]
            await set_database_global_context_variables(dataset.id, dataset.owner_id)

    graph_engine = await get_graph_engine()
    graph_data = await graph_engine.get_graph_data()

    graph = await cognee_network_visualization(graph_data, destination_file_path)

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
