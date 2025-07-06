import os
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.llm.structured_output_framework.baml.baml_client.async_client import b
from cognee.infrastructure.llm.structured_output_framework.baml_src.config import get_llm_config
from cognee.shared.data_models import SummarizedCode
from cognee.tasks.summarization.mock_summary import get_mock_summarized_code
from cognee.shared.logging_utils import get_logger
from instructor.exceptions import InstructorRetryException

logger = get_logger("extract_summary_baml")


async def extract_summary(content: str, response_model: Type[BaseModel]):
    """
    Extract summary using BAML framework.
    
    Args:
        content: The content to summarize
        response_model: The Pydantic model type for the response
        
    Returns:
        BaseModel: The summarized content in the specified format
    """
    config = get_llm_config()
    
    # Use BAML's SummarizeContent function
    summary_result = await b.SummarizeContent(content, baml_options={"tb": config.baml_registry})
    
    # Convert BAML result to the expected response model
    if response_model is SummarizedCode:
        # If it's asking for SummarizedCode but we got SummarizedContent, 
        # we need to use SummarizeCode instead
        code_result = await b.SummarizeCode(content, baml_options={"tb": config.baml_registry})
        return code_result
    else:
        # For other models, return the summary result
        return summary_result


async def extract_code_summary(content: str):
    """
    Extract code summary using BAML framework with mocking support.
    
    Args:
        content: The code content to summarize
        
    Returns:
        SummarizedCode: The summarized code information
    """
    enable_mocking = os.getenv("MOCK_CODE_SUMMARY", "false")
    if isinstance(enable_mocking, bool):
        enable_mocking = str(enable_mocking).lower()
    enable_mocking = enable_mocking in ("true", "1", "yes")

    if enable_mocking:
        result = get_mock_summarized_code()
        return result
    else:
        try:
            config = get_llm_config()
            result = await b.SummarizeCode(content, baml_options={"tb": config.baml_registry})
        except Exception as e:
            logger.error("Failed to extract code summary with BAML, falling back to mock summary", exc_info=e)
            result = get_mock_summarized_code()

        return result
