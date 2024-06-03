import logging
from cognee.modules.topology.extraction.extract_topology import extract_topology
from cognee.infrastructure.databases.graph.config import get_graph_config

logger = logging.getLogger(__name__)

async def infer_data_topology(content: str, graph_topology=None):
    if graph_topology is None:
        graph_config = get_graph_config()
        graph_topology = graph_config.graph_model
    try:
        return (await extract_topology(
            content,
            graph_topology
        ))
    except Exception as error:
        logger.error("Error extracting cognitive layers from content: %s", error, exc_info = True)
        raise error
