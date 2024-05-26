import logging
from .extraction.extract_categories import extract_categories
from cognee.modules.cognify.config import get_cognify_config

config = get_cognify_config()
logger = logging.getLogger(__name__)

async def get_content_categories(content: str):
    try:
        return await extract_categories(
            content,
            config.classification_model
        )
    except Exception as error:
        logger.error("Error extracting categories from content: %s", error, exc_info = True)
        raise error
