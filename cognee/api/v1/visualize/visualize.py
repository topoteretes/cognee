from cognee.shared.utils import create_cognee_style_network_with_logo, graph_to_tuple
from cognee.infrastructure.databases.graph import get_graph_engine
import logging


async def visualize_graph(label: str = "name"):
    """ """
    graph_engine = await get_graph_engine()
    graph_data = await graph_engine.get_graph_data()
    logging.info(graph_data)

    graph = await create_cognee_style_network_with_logo(graph_data, label=label)

    return graph
