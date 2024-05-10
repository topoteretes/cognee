import logging
from typing import List, Dict
from cognee.infrastructure import infrastructure_config
from cognee.modules.topology.extraction.extract_topology import extract_categories


logger = logging.getLogger(__name__)

async def infer_data_topology(content: str, graph_topology=None):
    if graph_topology is None:
        graph_topology = infrastructure_config.get_config()["graph_topology"]
    try:
        return (await extract_categories(
            content,
            graph_topology
        ))
    except Exception as error:
        logger.error("Error extracting cognitive layers from content: %s", error, exc_info = True)
        raise error
