import logging
from cognee.infrastructure import infrastructure_config
from.extraction.extract_summary import extract_summary
from cognee.modules.cognify.config import get_cognify_config

config = get_cognify_config()
logger = logging.getLogger(__name__)

async def get_content_summary(content: str):
    try:
        return await extract_summary(
            content,
            config.summarization_model
        )
    except Exception as error:
        logger.error("Error extracting summary from content: %s", error, exc_info = True)
        raise error
