import logging
from typing import List, Dict
from cognee.infrastructure import infrastructure_config
from.extraction.extract_cognitive_layers import extract_cognitive_layers

logger = logging.getLogger(__name__)

async def get_cognitive_layers(content: str, categories: List[Dict]):
    try:
        return (await extract_cognitive_layers(
            content,
            categories[0],
            infrastructure_config.get_config()["cognitive_layer_model"]
        )).cognitive_layers
    except Exception as error:
        logger.error("Error extracting cognitive layers from content: %s", error, exc_info = True)
        raise error
