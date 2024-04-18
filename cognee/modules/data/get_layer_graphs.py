import logging
import asyncio
from typing import List, Tuple
from cognee.infrastructure import infrastructure_config
from .extraction.extract_content_graph import extract_content_graph

logger = logging.getLogger(__name__)

async def get_layer_graphs(content: str, cognitive_layers: List[Tuple[str, dict]]):
    try:
        print("content: ", content)
        print("cognitive_layers: ", cognitive_layers)

        graph_awaitables = [
            extract_content_graph(
                content,
                cognitive_layer_data["name"],
                infrastructure_config.get_config()["graph_model"]
            ) for (_, cognitive_layer_data) in cognitive_layers
        ]

        graphs = await asyncio.gather(*graph_awaitables)

        return [(layer_id, graphs[layer_index]) for (layer_index, (layer_id, __)) in enumerate(cognitive_layers)]
    except Exception as error:
        logger.error("Error extracting graph from content: %s", error, exc_info = True)
        raise error
