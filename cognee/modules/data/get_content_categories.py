import logging
from cognee.infrastructure import infrastructure_config
from .extraction.extract_categories import extract_categories

logger = logging.getLogger(__name__)

async def get_content_categories(content: str):
    try:
        return await extract_categories(
            content,
            infrastructure_config.get_config()["classification_model"]
        )
    except Exception as error:
        logger.error("Error extracting categories from content: %s", error, exc_info = True)
        raise error
