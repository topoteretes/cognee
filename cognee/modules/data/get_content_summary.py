import logging
from cognee.infrastructure import infrastructure_config
from.extraction.extract_summary import extract_summary

logger = logging.getLogger(__name__)

async def get_content_summary(content: str):
    try:
        return await extract_summary(
            content,
            infrastructure_config.get_config()["summarization_model"]
        )
    except Exception as error:
        logger.error("Error extracting summary from content: %s", error, exc_info = True)
        raise error
