from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)
from cognee.infrastructure.databases.graph import get_graph_engine
import logging


import asyncio
from cognee.shared.utils import setup_logging


async def visualize_graph():
    graph_engine = await get_graph_engine()
    graph_data = await graph_engine.get_graph_data()
    logging.info(graph_data)

    graph = await cognee_network_visualization(graph_data)
    logging.info("The HTML file has been stored on your home directory! Navigate there with cd ~")

    return graph


if __name__ == "__main__":
    setup_logging(logging.ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(visualize_graph())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
