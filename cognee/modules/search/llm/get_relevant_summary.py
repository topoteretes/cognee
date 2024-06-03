import logging
from typing import List, Dict
from cognee.modules.cognify.config import get_cognify_config
from .extraction.categorize_relevant_summary import categorize_relevant_summary

logger = logging.getLogger(__name__)

async def get_cognitive_layers(content: str, categories: List[Dict]):
    try:
        cognify_config = get_cognify_config()
        return (await categorize_relevant_summary(
            content,
            categories[0],
            cognify_config.summarization_model,
        )).cognitive_layers
    except Exception as error:
        logger.error("Error extracting cognitive layers from content: %s", error, exc_info = True)
        raise error
